import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_analytics import results_to_csv, summary_to_csv, summarize_results


class TestAnalytics(unittest.TestCase):
    def test_summary_uses_only_successful_images(self):
        rows = [
            {"filename": "a.png", "status": "success", "predicted_class": "Forest", "top1_score": .8},
            {"filename": "b.png", "status": "success", "predicted_class": "Forest", "top1_score": .6},
            {"filename": "c.png", "status": "success", "predicted_class": "River", "top1_score": .9},
            {"filename": "d.png", "status": "error", "predicted_class": "", "top1_score": None},
        ]
        s = summarize_results(rows, ["Forest", "River", "Industrial"])
        self.assertEqual((s["total"], s["successful"], s["failed"]), (4, 3, 1))
        self.assertEqual([r["count"] for r in s["distribution"]], [2, 1, 0])
        self.assertAlmostEqual(s["distribution"][0]["percentage"], 200.0 / 3.0)
        self.assertAlmostEqual(s["mean_top1_score"], (0.8 + 0.6 + 0.9) / 3)

    def test_empty_summary(self):
        s = summarize_results([], ["Forest", "River"])
        self.assertEqual(s["successful"], 0)
        self.assertEqual(s["distribution"][0]["percentage"], 0)
        self.assertIsNone(s["mean_top1_score"])

    def test_csv_export_one_row_per_input(self):
        rows = [
            {"filename": "a.png", "model": "ResNet50 RGB", "status": "success", "predicted_class": "Forest", "top1_score": 0.9},
            {"filename": "bad.jpg", "model": "ResNet50 RGB", "status": "error", "error": "Invalid image"},
        ]
        decoded = results_to_csv(rows).decode("utf-8-sig")
        reader = list(csv.DictReader(io.StringIO(decoded)))
        self.assertEqual(len(reader), 2)
        self.assertEqual(reader[1]["status"], "error")
        self.assertEqual(reader[0]["filename"], "a.png")

    def test_csv_formula_injection_protection(self):
        decoded = results_to_csv([{
            "filename": "=HYPERLINK(\"https://bad.test\",\"click\")",
            "status": "error", "error": "+1+2",
        }]).decode("utf-8-sig")
        reader = list(csv.DictReader(io.StringIO(decoded)))
        self.assertTrue(reader[0]["filename"].startswith("'="))
        self.assertTrue(reader[0]["error"].startswith("'+"))

    def test_summary_csv(self):
        s = summarize_results([{"status": "success", "predicted_class": "Forest", "top1_score": .7}], ["Forest"])
        data = summary_to_csv(s).decode("utf-8-sig")
        self.assertIn("class,count,percentage", data)
        self.assertIn("Forest,1,100.0", data)


if __name__ == "__main__":
    unittest.main()
