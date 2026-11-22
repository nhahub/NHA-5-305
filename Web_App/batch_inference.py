"""Safe, bounded RGB batch inference using the existing EuroSAT checkpoints.

No training, new model weights, or changes to RGB normalization are required.
Streamlit UploadedFile-like values need .name and .getvalue().
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

import torch
from PIL import Image, UnidentifiedImageError

from inference import (
    MODEL_RGB_EFFICIENT,
    MODEL_RGB_RESNET,
    load_model,
    predict_tensors,
    preprocess_rgb,
)

MAX_IMAGES = 100
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_ZIP_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_COMPRESSION_RATIO = 200
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESULT_COLUMNS = (
    "filename", "model", "status", "predicted_class", "top1_score",
    "top2_class", "top2_score", "top3_class", "top3_score", "error",
)

ProgressCallback = Callable[[int, int], None]


def _empty_row(name: str, model: str) -> dict:
    return {
        "filename": name, "model": model, "status": "error", "predicted_class": "",
        "top1_score": None, "top2_class": "", "top2_score": None,
        "top3_class": "", "top3_score": None, "error": "",
    }


def _safe_archive_name(name: str) -> bool:
    """Disallow absolute, traversal, Windows-drive, and backslash ZIP entries."""
    normalized = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and not re.match(r"^[A-Za-z]:", name) and ".." not in normalized.parts


def _uploaded_items(files: Sequence, zip_upload=None) -> list[tuple[str, bytes | None, str | None]]:
    """Read bounded inputs. Return (filename, bytes, validation_error) in original order."""
    if zip_upload is not None and files:
        raise ValueError("Choose either multiple image files or one ZIP archive, not both.")
    if zip_upload is None:
        if not files:
            raise ValueError("Upload one or more JPG/PNG images first.")
        if len(files) > MAX_IMAGES:
            raise ValueError(f"Maximum {MAX_IMAGES} images per job.")
        items = []
        total = 0
        for f in files:
            name = str(getattr(f, "name", "unnamed"))
            size = getattr(f, "size", None)
            # Apply bounds before getvalue() where metadata is available.
            if isinstance(size, int) and size > MAX_IMAGE_BYTES:
                items.append((name, None, "Image exceeds the 10 MiB per-file limit"))
                continue
            content = f.getvalue()
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Total upload exceeds the 100 MiB batch limit.")
            if len(content) > MAX_IMAGE_BYTES:
                items.append((name, None, "Image exceeds the 10 MiB per-file limit"))
            else:
                items.append((name, content, None))
        return items

    size = getattr(zip_upload, "size", None)
    if isinstance(size, int) and size > MAX_ZIP_BYTES:
        raise ValueError("ZIP exceeds the 50 MiB compressed upload limit.")
    content = zip_upload.getvalue()
    if len(content) > MAX_ZIP_BYTES:
        raise ValueError("ZIP exceeds the 50 MiB compressed upload limit.")
    items = []
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
            entries = [info for info in archive.infolist() if not info.is_dir()]
            if not entries:
                raise ValueError("ZIP archive contains no files.")
            if len(entries) > MAX_IMAGES:
                raise ValueError(f"ZIP may contain at most {MAX_IMAGES} files.")
            total_uncompressed = 0
            for info in entries:
                if not _safe_archive_name(info.filename):
                    raise ValueError("ZIP contains an unsafe file path.")
                if info.flag_bits & 0x1:
                    raise ValueError("Encrypted ZIP entries are not supported.")
                if info.file_size > MAX_IMAGE_BYTES:
                    items.append((info.filename, None, "Image exceeds the 10 MiB per-file limit"))
                    continue
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_TOTAL_BYTES:
                    raise ValueError("ZIP exceeds the 100 MiB uncompressed batch limit.")
                if info.file_size > 0 and (info.compress_size == 0 or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO):
                    raise ValueError("Suspicious ZIP compression ratio detected.")
                if Path(info.filename).suffix.lower() not in VALID_EXTENSIONS:
                    items.append((info.filename, None, "Unsupported file extension"))
                    continue
                with archive.open(info) as reader:
                    data = reader.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    items.append((info.filename, None, "Image exceeds the 10 MiB per-file limit"))
                else:
                    items.append((info.filename, data, None))
    except (zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("Invalid or damaged ZIP archive.") from exc
    return items


def _read_rgb(name: str, data: bytes | None) -> Image.Image:
    if Path(name).suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError("Only JPG and PNG images are supported")
    if not data:
        raise ValueError("Empty image file")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 10 MiB per-file limit")
    try:
        with Image.open(io.BytesIO(data)) as src:
            if src.format not in {"JPEG", "PNG"}:
                raise ValueError("Image content is not JPG or PNG")
            if src.width * src.height > MAX_PIXELS:
                raise ValueError("Image dimensions exceed 16 million pixels")
            src.load()
            return src.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Unreadable or unsafe image") from exc


def classify_rgb_batch(
    files: Sequence,
    model_name: str,
    batch_size: int = 8,
    *,
    zip_upload=None,
    on_progress: ProgressCallback | None = None,
) -> list[dict]:
    """Return one ordered result per uploaded image, including failures.

    Model loads once per job, tensors are materialized only per batch. Uses
    inference.preprocess_rgb and inference.predict_tensors: no retraining.
    """
    if model_name not in (MODEL_RGB_RESNET, MODEL_RGB_EFFICIENT):
        raise ValueError("Batch processing currently supports RGB models only.")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
        raise ValueError("Batch size must be between 1 and 16.")
    items = _uploaded_items(files, zip_upload)
    rows: list[dict | None] = [None] * len(items)
    pending: list[tuple[int, torch.Tensor]] = []
    model = checkpoint = None
    processed = 0

    def report():
        if on_progress is not None:
            on_progress(processed, len(items))

    def flush():
        nonlocal pending, processed
        if not pending:
            return
        batch = torch.cat([tensor for _, tensor in pending], dim=0)
        try:
            predictions = predict_tensors(model, checkpoint, batch)
            if len(predictions) != len(pending):
                raise ValueError("Prediction count does not match the input batch")
            for (idx, _), prediction in zip(pending, predictions):
                row = rows[idx]
                row.update({
                    "status": "success",
                    "predicted_class": prediction["prediction"],
                    "top1_score": prediction["top3"][0]["score"],
                    "top2_class": prediction["top3"][1]["class_name"],
                    "top2_score": prediction["top3"][1]["score"],
                    "top3_class": prediction["top3"][2]["class_name"],
                    "top3_score": prediction["top3"][2]["score"],
                })
        except (ValueError, RuntimeError, OSError) as exc:
            # Preserve a report row for every input rather than crashing the whole job.
            for idx, _ in pending:
                rows[idx]["error"] = f"Model inference failed: {str(exc)[:180]}"
        processed += len(pending)
        pending = []
        report()

    for idx, (filename, data, validation_error) in enumerate(items):
        rows[idx] = _empty_row(filename, model_name)
        if validation_error:
            rows[idx]["error"] = validation_error
            processed += 1
            report()
            continue
        try:
            image = _read_rgb(filename, data)
            if model is None:
                model, checkpoint = load_model(model_name)
            tensor = preprocess_rgb(image, checkpoint)
            pending.append((idx, tensor))
            if len(pending) == batch_size:
                flush()
        except (ValueError, TypeError, RuntimeError, OSError, FileNotFoundError) as exc:
            rows[idx]["error"] = str(exc)[:240]
            processed += 1
            report()
    flush()
    return rows
