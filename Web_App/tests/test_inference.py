"""Offline tests; original SHA256-verified checkpoints; no dataset/training required."""
import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference as inf


class FakeStreamlit(types.ModuleType):
    """Tiny stub for testing Streamlit app callback/wiring code offline.

    This is NOT a replacement for running the actual Streamlit server in deployment.
    """
    def __init__(self):
        super().__init__("streamlit")
        self.events = []
        self.cache_data = lambda **kwargs: (lambda f: f)

    def __getattr__(self, name):
        if name in ("sidebar",):
            return self
        def method(*args, **kwargs):
            self.events.append((name, args))
            if name in ("form", "expander"):
                return self
            if name in ("tabs", "columns"):
                return [self for _ in (args[0] if isinstance(args[0], (tuple, list)) else range(args[0]))]
            if name == "selectbox":
                return args[1][0]
            if name == "form_submit_button":
                return False
            if name == "file_uploader":
                return None
            if name == "checkbox":
                return False
            return None
        return method

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Upload:
    def __init__(self, filename, content):
        self.name = filename
        self._content = content
    def getvalue(self):
        return self._content


class CheckpointAndFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(123)
        cls.rgb = Image.fromarray(cls.rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))

    @staticmethod
    def write_ms(path, array, descriptions=None):
        with rasterio.open(path, "w", driver="GTiff", count=array.shape[0],
                           height=array.shape[1], width=array.shape[2],
                           dtype="float32", transform=from_origin(0, 0, 10, 10)) as dst:
            dst.write(array)
            if descriptions:
                for i, description in enumerate(descriptions, 1):
                    dst.set_band_description(i, description)

    def test_manifest_sha256_and_real_models(self):
        self.assertEqual(len(inf.EXPECTED_MODELS), 3)
        for name in inf.EXPECTED_MODELS:
            self.assertEqual(inf.sha256_file(inf.validate_checkpoint_file(name)),
                             inf.model_info(name)["sha256"])
            model, ckpt = inf.load_model(name)
            self.assertEqual(len(ckpt["class_names"]), 10)
            self.assertFalse(model.training)
            self.assertNotIn("model_state_dict", ckpt, "Duplicated tensors should be released to save cloud RAM")
        # Cache only the most recently requested model.
        self.assertLessEqual(inf.load_model.cache_info().currsize, 1)

    def test_rgb_real_checkpoint_inference(self):
        for name in (inf.MODEL_RGB_RESNET, inf.MODEL_RGB_EFFICIENT):
            with self.subTest(name=name):
                model, ckpt = inf.load_model(name)
                batch = inf.preprocess_rgb(self.rgb, ckpt)
                self.assertEqual(tuple(batch.shape), (1, 3, 224, 224))
                result = inf.predict_rgb(self.rgb, name)
                self.assertEqual(len(result["top3"]), 3)
                with torch.inference_mode():
                    independent = torch.softmax(model(batch), dim=1).max().item()
                self.assertAlmostEqual(result["model_score"], independent, places=5)

    def test_multispectral_real_prediction_and_band_reordering(self):
        _, ckpt = inf.load_model(inf.MODEL_MS)
        array = np.empty((13, 64, 64), dtype=np.float32)
        for i, mean in enumerate(ckpt["normalization_mean"]):
            array[i] = mean + self.rng.normal(0, 20, (64, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            ordered = Path(tmp, "ordered.tif")
            reversed_ = Path(tmp, "reversed.tiff")
            wrong_count = Path(tmp, "wrong.tif")
            self.write_ms(ordered, array)
            order = list(range(12, -1, -1))
            self.write_ms(reversed_, array[order], [inf.BAND_ORDER[i] for i in order])
            self.write_ms(wrong_count, array[:3])
            with self.assertRaisesRegex(ValueError, "band names"):
                inf.predict_ms(ordered)
            with self.assertRaisesRegex(ValueError, "exactly 13"):
                inf.predict_ms(wrong_count, confirm_unlabeled_band_order=True)
            tensor_0 = inf.preprocess_ms(ordered, ckpt, confirm_unlabeled_band_order=True)
            tensor_1 = inf.preprocess_ms(reversed_, ckpt)
            self.assertEqual(tuple(tensor_1.shape), (1, 13, 224, 224))
            self.assertTrue(torch.allclose(tensor_0, tensor_1, atol=1e-6))
            self.assertEqual(inf.predict_ms(reversed_)["prediction"],
                             inf.predict_ms(ordered, confirm_unlabeled_band_order=True)["prediction"])

    def test_invalid_rgb_and_model_selections_rejected(self):
        with self.assertRaisesRegex(ValueError, "Upload"):
            inf.predict_rgb(None, inf.MODEL_RGB_RESNET)
        with self.assertRaisesRegex(ValueError, "Select an RGB"):
            inf.predict_rgb(self.rgb, inf.MODEL_MS)

    def test_streamlit_app_source_wiring_and_upload_guardrails(self):
        fake = FakeStreamlit()
        with patch.dict(sys.modules, {"streamlit": fake}):
            sys.modules.pop("app", None)
            app = importlib.import_module("app")
            app.main()
            tabs = [event for event in fake.events if event[0] == "tabs"]
            self.assertEqual(len(tabs), 1)
            self.assertEqual(len(tabs[0][1][0]), 3)
            self.assertEqual(sum(1 for name, _ in fake.events if name == "file_uploader"), 2)
            self.assertEqual(sum(1 for name, _ in fake.events if name == "form_submit_button"), 2)
            buf = io.BytesIO()
            self.rgb.save(buf, format="PNG")
            loaded = app.load_rgb_upload(Upload("satellite.png", buf.getvalue()))
            self.assertEqual(loaded.mode, "RGB")
            with self.assertRaisesRegex(ValueError, "readable JPG or PNG"):
                app.load_rgb_upload(Upload("wrong.png", b"not an image"))
            with self.assertRaisesRegex(ValueError, "Confirm the sensor"):
                app.classify_ms_upload(Upload("test.tif", b"hi"), False)
            with self.assertRaisesRegex(ValueError, "maximum 25 MB"):
                app.classify_ms_upload(Upload("test.tif", b"0" * (25*1024*1024+1)), True)
            sys.modules.pop("app", None)

    def test_git_ready_metadata(self):
        repo = ROOT.parent
        self.assertIn("filter=lfs", (repo / ".gitattributes").read_text())
        self.assertIn("headless", (repo / ".streamlit" / "config.toml").read_text())
        self.assertTrue((repo / "Land_Type_Classification_Final.ipynb").exists())
        req = (ROOT / "requirements.txt").read_text()
        self.assertIn("streamlit", req)
        self.assertNotIn("gradio==", req)
        self.assertIn("torch==2.10.0+cpu", req)
        self.assertIn("torchvision==0.25.0+cpu", req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
