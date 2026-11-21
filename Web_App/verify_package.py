"""Run: python verify_package.py — validates real .pth weights and smoke predictions."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from inference import (
    EXPECTED_MODELS, MODEL_MS, MODEL_RGB_EFFICIENT, MODEL_RGB_RESNET,
    get_manifest, load_model, predict_ms, predict_rgb, verify_all_checkpoints,
)


def make_test_tiff(path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    # Test input ONLY: random data verifies wiring, not geospatial accuracy.
    arr = np.empty((13, 64, 64), dtype=np.uint16)
    means = np.asarray(load_model(MODEL_MS)[1]["normalization_mean"])
    for i, mean in enumerate(means):
        arr[i] = np.uint16(max(1, mean))
    with rasterio.open(path, "w", driver="GTiff", count=13, height=64, width=64,
                       dtype="uint16", transform=from_origin(0, 0, 10, 10)) as dst:
        dst.write(arr)


def main() -> None:
    print("Verifying actual checkpoint files against the supplied SHA256 manifest...")
    assert len(EXPECTED_MODELS) == 3
    for label, status in verify_all_checkpoints().items():
        print(label, status)
    rng = np.random.default_rng(42)
    rgb = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), mode="RGB")
    for name in (MODEL_RGB_RESNET, MODEL_RGB_EFFICIENT):
        result = predict_rgb(rgb, name)
        assert result["prediction"] in get_manifest()["classes"]
        assert len(result["top3"]) == 3
        print("RGB inference PASS:", name, result["prediction"])
    with tempfile.TemporaryDirectory() as folder:
        tiff_path = Path(folder) / "smoke_test.tif"
        make_test_tiff(tiff_path)
        try:
            predict_ms(tiff_path)
        except ValueError as exc:
            assert "band names" in str(exc)
            print("MS TIFF metadata-safety validation PASS")
        else:
            raise AssertionError("Unlabeled TIFF was not blocked")
        result = predict_ms(tiff_path, confirm_unlabeled_band_order=True)
        assert result["prediction"] in get_manifest()["classes"]
        assert len(result["top3"]) == 3
        print("MS inference PASS:", result["prediction"])
    print("PASSED: all 3 genuine checkpoints load; synthetic RGB/TIFF inference succeeds.")
    print("NOTE: Synthetic smoke tests do not measure classification accuracy.")


if __name__ == "__main__":
    main()
