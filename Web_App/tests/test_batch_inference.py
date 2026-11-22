"""Batch tests with a deterministic mock network (no actual checkpoint required)."""
import io
import os
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

import inference
import batch_inference as batch


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.size = len(content)

    def getvalue(self):
        return self.content


def image_bytes(color, fmt="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(stream, format=fmt)
    return stream.getvalue()


class FakeNet(torch.nn.Module):
    def forward(self, tensor):
        avg = tensor.mean(dim=(2, 3))
        return torch.stack([avg[:, i % 3] * (0.3 + i * .03) + i * .05 for i in range(10)], dim=1)


META = {
    "input_channels": 3,
    "image_size": 32,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "class_names": [f"Class{i}" for i in range(10)],
}


class TestBatchInference(unittest.TestCase):
    def setUp(self):
        self.model = FakeNet().eval()
        self.model_label = inference.MODEL_RGB_RESNET

    def run_images(self, files, *, batch_size=8, zip_upload=None, on_progress=None):
        with patch("batch_inference.load_model", return_value=(self.model, META)) as mocked:
            rows = batch.classify_rgb_batch(
                files, self.model_label, batch_size,
                zip_upload=zip_upload, on_progress=on_progress,
            )
        return rows, mocked.call_count

    def test_17_images_across_three_batches_and_single_equivalence(self):
        files = [FakeUpload(f"image_{i:03}.png", image_bytes((i * 5, i * 7, i * 11))) for i in range(17)]
        rows, loads = self.run_images(files, batch_size=8)
        self.assertEqual(loads, 1)
        self.assertEqual(len(rows), 17)
        self.assertTrue(all(r["status"] == "success" for r in rows))
        for file, row in zip(files, rows):
            img = Image.open(io.BytesIO(file.getvalue()))
            expected = inference.predict_tensor(self.model, META, inference.preprocess_rgb(img, META))
            self.assertEqual(row["predicted_class"], expected["prediction"])
            self.assertAlmostEqual(row["top1_score"], expected["model_score"], places=6)

    def test_corrupted_image_does_not_stop_job(self):
        files = [
            FakeUpload("a.jpg", image_bytes((30, 20, 10), "JPEG")),
            FakeUpload("broken.png", b"not a png"),
            FakeUpload("c.png", image_bytes((40, 50, 60))),
        ]
        rows, _ = self.run_images(files, batch_size=2)
        self.assertEqual([r["status"] for r in rows], ["success", "error", "success"])
        self.assertEqual([r["filename"] for r in rows], [f.name for f in files])
        self.assertIn("Unreadable", rows[1]["error"])

    def test_wrong_extension_rejected_per_file(self):
        files = [FakeUpload("a.txt", image_bytes((20, 30, 40)))]
        rows, loads = self.run_images(files)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(loads, 0)

    def test_empty_upload_rejected(self):
        with self.assertRaisesRegex(ValueError, "Upload one or more"):
            batch.classify_rgb_batch([], self.model_label)

    def test_too_many_files_rejected(self):
        files = [FakeUpload(f"{i}.png", b"") for i in range(batch.MAX_IMAGES + 1)]
        with self.assertRaisesRegex(ValueError, "Maximum"):
            batch.classify_rgb_batch(files, self.model_label)

    def test_out_of_range_batch_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "Batch size"):
            batch.classify_rgb_batch([FakeUpload("x.png", b"1")], self.model_label, batch_size=0)

    def test_multispectral_model_not_allowed(self):
        with self.assertRaisesRegex(ValueError, "RGB"):
            batch.classify_rgb_batch([FakeUpload("x.png", b"1")], inference.MODEL_MS)

    def test_progress_finishes_at_total(self):
        progress = []
        files = [FakeUpload("a.png", image_bytes((5, 5, 5))), FakeUpload("broken.png", b"broken")]
        self.run_images(files, batch_size=1, on_progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(progress[-1], (2, 2))
        self.assertTrue(all(0 <= done <= total for done, total in progress))

    def test_zip_archive_support(self):
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("forest/a.png", image_bytes((10, 100, 20)))
            zf.writestr("other/b.jpg", image_bytes((30, 60, 90), "JPEG"))
            zf.writestr("notes.txt", "unsupported")
        upload = FakeUpload("images.zip", archive_bytes.getvalue())
        rows, loads = self.run_images([], zip_upload=upload)
        self.assertEqual(loads, 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["status"] for r in rows], ["success", "success", "error"])
        self.assertEqual(rows[0]["filename"], "forest/a.png")

    def test_zip_path_traversal_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../escape.png", image_bytes((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "unsafe file path"):
            batch.classify_rgb_batch([], self.model_label, zip_upload=FakeUpload("bad.zip", stream.getvalue()))

    def test_zip_oversized_declared_entry_rejected_without_decompression(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("huge.png", b"A" * (batch.MAX_IMAGE_BYTES + 1))
        rows, loads = self.run_images([], zip_upload=FakeUpload("huge.zip", stream.getvalue()))
        self.assertEqual(loads, 0)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("10 MiB", rows[0]["error"])

    def test_zip_bomb_ratio_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("compressed.png", b"A" * 100000)
        with self.assertRaisesRegex(ValueError, "compression ratio"):
            batch.classify_rgb_batch([], self.model_label, zip_upload=FakeUpload("zip.zip", stream.getvalue()))

    def test_cannot_mix_upload_and_zip(self):
        with self.assertRaisesRegex(ValueError, "either"):
            batch.classify_rgb_batch(
                [FakeUpload("a.png", image_bytes((1, 2, 3)))], self.model_label,
                zip_upload=FakeUpload("x.zip", b"zip"),
            )


    def test_comparison_runs_both_models_once_and_preserves_filename_alignment(self):
        class FixedClassNet(torch.nn.Module):
            def __init__(self, class_index):
                super().__init__()
                self.class_index = class_index

            def forward(self, tensor):
                logits = torch.zeros((tensor.shape[0], 10), dtype=tensor.dtype)
                logits[:, self.class_index] = 5.0
                return logits

        files = [
            FakeUpload("first.png", image_bytes((10, 20, 30))),
            FakeUpload("second.png", image_bytes((40, 50, 60))),
            FakeUpload("third.png", image_bytes((70, 80, 90))),
        ]
        progress = []
        with patch(
            "batch_inference.load_model",
            side_effect=[(FixedClassNet(1).eval(), META), (FixedClassNet(2).eval(), META)],
        ) as mocked:
            rows = batch.classify_rgb_batch_comparison(
                files, batch_size=2,
                on_progress=lambda done, total: progress.append((done, total)),
            )

        self.assertEqual(mocked.call_count, 2)
        mocked.cache_clear.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in mocked.call_args_list],
            [inference.MODEL_RGB_RESNET, inference.MODEL_RGB_EFFICIENT],
        )
        self.assertEqual([row["filename"] for row in rows], [file.name for file in files])
        self.assertEqual([row["comparison_status"] for row in rows], ["disagreement"] * 3)
        self.assertTrue(all(row["review_required"] for row in rows))
        self.assertEqual(progress[-1], (6, 6))

    def test_comparison_keeps_partial_model_failure(self):
        files = [
            FakeUpload("one.png", image_bytes((10, 20, 30))),
            FakeUpload("two.png", image_bytes((40, 50, 60))),
        ]
        with patch(
            "batch_inference.load_model",
            side_effect=[(FakeNet().eval(), META), OSError("EfficientNet checkpoint unavailable")],
        ) as mocked:
            rows = batch.classify_rgb_batch_comparison(files, batch_size=1)

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual([row["comparison_status"] for row in rows], ["partial_failure"] * 2)
        self.assertTrue(all(row["resnet50_status"] == "success" for row in rows))
        self.assertTrue(all(row["efficientnet_status"] == "error" for row in rows))
        self.assertTrue(all(row["review_required"] for row in rows))
        self.assertTrue(all(row["review_priority"] == "high" for row in rows))

    def test_comparison_corrupt_image_isolated_and_progress_completes(self):
        files = [
            FakeUpload("good.png", image_bytes((10, 20, 30))),
            FakeUpload("broken.png", b"not an image"),
            FakeUpload("also_good.png", image_bytes((40, 50, 60))),
        ]
        progress = []
        with patch(
            "batch_inference.load_model",
            side_effect=[(FakeNet().eval(), META), (FakeNet().eval(), META)],
        ):
            rows = batch.classify_rgb_batch_comparison(
                files, batch_size=2,
                on_progress=lambda done, total: progress.append((done, total)),
            )
        self.assertEqual([row["filename"] for row in rows], [f.name for f in files])
        self.assertEqual(rows[0]["comparison_status"], "agreement")
        self.assertEqual(rows[1]["comparison_status"], "failed")
        self.assertEqual(rows[2]["comparison_status"], "agreement")
        self.assertTrue(rows[1]["review_required"])
        self.assertEqual(progress[-1], (6, 6))

    def test_comparison_rejects_invalid_threshold(self):
        files = [FakeUpload("one.png", image_bytes((10, 20, 30)))]
        with self.assertRaisesRegex(ValueError, "Low-score threshold"):
            batch.classify_rgb_batch_comparison(files, low_score_threshold=2.0)



if __name__ == "__main__":
    unittest.main()
