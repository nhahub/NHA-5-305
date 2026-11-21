"""Checkpoint-only EuroSAT inference: no training data, notebook, or network required.

The 13-channel TIFF preprocessing mirrors the executed final training notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
MANIFEST_PATH = ROOT / "deployment_manifest.json"

MODEL_RGB_RESNET = "ResNet50 RGB"
MODEL_RGB_EFFICIENT = "EfficientNet-B0 RGB"
MODEL_MS = "ResNet50 Multispectral"
EXPECTED_MODELS = (MODEL_RGB_RESNET, MODEL_RGB_EFFICIENT, MODEL_MS)
BAND_ORDER = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B09", "B10", "B11", "B12", "B8A",
)

# Limit PyTorch to the modest CPU allocation available on free hosting.
torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))


@lru_cache(maxsize=1)
def get_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("models"), dict):
        raise ValueError("deployment_manifest.json is missing its models section")
    if set(manifest["models"]) != set(EXPECTED_MODELS):
        raise ValueError("The manifest must contain exactly the three trained EuroSAT models")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_info(model_label: str) -> dict[str, Any]:
    if model_label not in EXPECTED_MODELS:
        raise ValueError(f"Unknown model selection: {model_label}")
    return get_manifest()["models"][model_label]


def checkpoint_path(model_label: str) -> Path:
    filename = model_info(model_label).get("filename")
    if not isinstance(filename, str) or filename != Path(filename).name:
        raise ValueError("Invalid checkpoint filename in manifest")
    return CHECKPOINT_DIR / filename


def validate_checkpoint_file(model_label: str) -> Path:
    details = model_info(model_label)
    path = checkpoint_path(model_label)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.name}")
    if sha256_file(path).lower() != str(details["sha256"]).lower():
        raise ValueError(f"Checkpoint integrity check failed: {path.name} (SHA256 mismatch)")
    return path


def build_model(architecture: str, input_channels: int, num_classes: int) -> nn.Module:
    if architecture == "resnet50" and input_channels == 3:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0" and input_channels == 3:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "resnet50_multispectral" and input_channels == 13:
        model = models.resnet50(weights=None)
        old_stem = model.conv1
        model.conv1 = nn.Conv2d(
            13, old_stem.out_channels, kernel_size=old_stem.kernel_size,
            stride=old_stem.stride, padding=old_stem.padding, bias=False,
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported checkpoint architecture/channels: {architecture}/{input_channels}")
    return model


@lru_cache(maxsize=1)
def load_model(model_label: str) -> tuple[nn.Module, dict[str, Any]]:
    """Verify SHA256 and lazily load one CPU model at a time (low-memory hosting)."""
    path = validate_checkpoint_file(model_label)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    info = model_info(model_label)
    for key in ("architecture", "input_channels", "image_size", "class_names", "model_state_dict"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint {path.name} missing {key}")
    for key in ("architecture", "input_channels", "image_size"):
        if checkpoint[key] != info[key]:
            raise ValueError(f"Manifest/checkpoint mismatch in {key}")
    if list(checkpoint["class_names"]) != list(get_manifest()["classes"]):
        raise ValueError("Class label order differs from verified training run")
    if len(checkpoint["class_names"]) != 10:
        raise ValueError("Expected ten EuroSAT classes")
    if int(checkpoint["input_channels"]) == 13:
        if list(checkpoint.get("band_order", [])) != list(BAND_ORDER):
            raise ValueError("The multispectral band order differs from the training contract")
        if not all(len(checkpoint.get(key, [])) == 13 for key in ("normalization_mean", "normalization_std")):
            raise ValueError("Multispectral checkpoint missing 13-channel normalization")
    else:
        if not all(len(checkpoint.get(key, [])) == 3 for key in ("imagenet_mean", "imagenet_std")):
            raise ValueError("RGB checkpoint missing ImageNet normalization")

    model = build_model(
        str(checkpoint["architecture"]), int(checkpoint["input_channels"]),
        len(checkpoint["class_names"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # The model now owns the weights. Drop duplicated tensors from the saved payload
    # to reduce RAM usage on Streamlit Community Cloud.
    del checkpoint["model_state_dict"]
    model.eval()
    return model, checkpoint


def preprocess_rgb(image: Image.Image, checkpoint: dict[str, Any]) -> torch.Tensor:
    if image is None:
        raise ValueError("Upload a JPG or PNG satellite image first")
    if not isinstance(image, Image.Image):
        raise TypeError("RGB input must be a Pillow image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image is empty")
    if image.width * image.height > 16_000_000:
        raise ValueError("Image is too large; please use a satellite patch")
    if int(checkpoint["input_channels"]) != 3:
        raise ValueError("Selected model needs a 13-band TIFF, not an RGB image")
    # Exact eval preprocessing from final notebook (including PIL Resize interpolation).
    pipeline = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"]))),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=list(checkpoint["imagenet_mean"]),
            std=list(checkpoint["imagenet_std"]),
        ),
    ])
    return pipeline(image.convert("RGB")).unsqueeze(0)


def _parse_band_name(description: str | None) -> str | None:
    if not description:
        return None
    # Match B01..B12 or B8A only; avoid accidentally recognizing B08 in B08A.
    match = re.search(r"(?<![A-Z0-9])B(?:0[1-9]|1[0-2]|8A)(?![A-Z0-9])", str(description).upper())
    return match.group(0) if match else None


def preprocess_ms(tiff_path: str | Path, checkpoint: dict[str, Any], *,
                  confirm_unlabeled_band_order: bool = False) -> torch.Tensor:
    """Mirror final notebook: re-order named bands, handle nodata, normalize, bilinear upsample."""
    if int(checkpoint["input_channels"]) != 13:
        raise ValueError("Selected model needs an RGB image, not a multispectral TIFF")
    file_path = Path(tiff_path) if tiff_path else None
    if file_path is None or not file_path.is_file():
        raise ValueError("Upload a 13-band TIFF file first")
    if file_path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("Expected a .tif or .tiff file")
    if file_path.stat().st_size > 200 * 1024 * 1024:
        raise ValueError("TIFF is too large; upload a small satellite patch")

    import rasterio
    with rasterio.open(file_path) as src:
        if src.count != 13:
            raise ValueError(f"Expected exactly 13 spectral bands; received {src.count}")
        if src.width * src.height > 1_048_576:
            raise ValueError("TIFF exceeds 1024×1024 pixels; crop it to a satellite patch")
        data = src.read(out_dtype="float32")
        masks = src.read_masks()
        nodata_values = list(src.nodatavals)
        descriptions = list(src.descriptions)

    expected_order = list(checkpoint["band_order"])
    descriptions_to_bands = [_parse_band_name(desc) for desc in descriptions]
    if all(descriptions_to_bands):
        if set(descriptions_to_bands) != set(expected_order) or len(set(descriptions_to_bands)) != 13:
            raise ValueError("TIFF band names differ from the EuroSAT 13-band order")
        reorder = [descriptions_to_bands.index(name) for name in expected_order]
        data, masks = data[reorder], masks[reorder]
        nodata_values = [nodata_values[idx] for idx in reorder]
    elif any(descriptions_to_bands):
        raise ValueError("TIFF has incomplete band names; band order cannot be verified")
    elif not confirm_unlabeled_band_order:
        raise ValueError(
            "This TIFF has no band names. Confirm that its band order is B01–B12, B8A "
            "and its values use the same Sentinel-2 scale as the EuroSAT training data."
        )

    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)[:, None, None]
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Invalid multispectral normalization parameters")
    valid = np.isfinite(data) & (masks > 0)
    for i, nodata in enumerate(nodata_values):
        if nodata is not None and np.isfinite(nodata):
            valid[i] &= data[i] != nodata
    # Mirror the source notebook's special handling for all-zero pixels.
    valid &= ~np.all(data == 0, axis=0)[None, :, :]
    clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    clean[~valid] = np.broadcast_to(mean, clean.shape)[~valid]
    normalized = (clean - mean) / std
    normalized[~valid] = 0.0
    tensor = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).unsqueeze(0)
    return F.interpolate(
        tensor, size=(int(checkpoint["image_size"]), int(checkpoint["image_size"])),
        mode="bilinear", align_corners=False,
    )


def predict_tensor(model: nn.Module, checkpoint: dict[str, Any], tensor: torch.Tensor) -> dict[str, Any]:
    channels = int(checkpoint["input_channels"])
    size = int(checkpoint["image_size"])
    if tensor.ndim != 4 or tuple(tensor.shape) != (1, channels, size, size):
        raise ValueError(f"Expected tensor shape (1, {channels}, {size}, {size})")
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    if not torch.isfinite(probabilities).all():
        raise ValueError("Model returned a non-finite prediction")
    scores, indexes = torch.topk(probabilities, k=3)
    names = checkpoint["class_names"]
    top3 = [
        {"class_name": names[int(i)], "score": float(score)}
        for score, i in zip(scores.tolist(), indexes.tolist())
    ]
    return {
        "prediction": top3[0]["class_name"],
        "model_score": top3[0]["score"],
        "top3": top3,
    }


def predict_rgb(image: Image.Image, model_label: str) -> dict[str, Any]:
    if model_label not in (MODEL_RGB_RESNET, MODEL_RGB_EFFICIENT):
        raise ValueError("Select an RGB model")
    model, checkpoint = load_model(model_label)
    return predict_tensor(model, checkpoint, preprocess_rgb(image, checkpoint))


def predict_ms(tiff_path: str | Path, *, confirm_unlabeled_band_order: bool = False) -> dict[str, Any]:
    model, checkpoint = load_model(MODEL_MS)
    return predict_tensor(
        model, checkpoint,
        preprocess_ms(tiff_path, checkpoint,
                      confirm_unlabeled_band_order=confirm_unlabeled_band_order),
    )


def verify_all_checkpoints() -> dict[str, str]:
    """Check verified model weights and metadata; does not re-train."""
    report = {}
    for name in EXPECTED_MODELS:
        model, ckpt = load_model(name)
        report[name] = (
            f"OK: {ckpt['architecture']}, {ckpt['input_channels']} channels, "
            f"{len(ckpt['class_names'])} classes, {sum(p.numel() for p in model.parameters()):,} parameters"
        )
    return report
