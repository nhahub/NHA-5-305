import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_review import (
    compare_model_results,
    comparison_results_to_csv,
    review_queue_to_csv,
    summarize_comparison_results,
)


def row(filename, status="success", prediction="Forest", top1=0.9, top2_class="Pasture", top2=0.05, error=""):
    return {
        "filename": filename,
        "status": status,
        "predicted_class": prediction if status == "success" else "",
        "top1_score": top1 if status == "success" else None,
        "top2_class": top2_class if status == "success" else "",
        "top2_score": top2 if status == "success" else None,
        "top3_class": "River" if status == "success" else "",
        "top3_score": 0.02 if status == "success" else None,
        "error": error,
    }


class TestSmartReview(unittest.TestCase):
    def test_agreement_is_not_automatically_reviewed(self):
        results = compare_model_results([row("a.png")], [row("a.png", top1=0.8)])
        self.assertEqual(results[0]["comparison_status"], "agreement")
        self.assertIs(results[0]["agreement"], True)
        self.assertFalse(results[0]["review_required"])
        self.assertEqual(results[0]["review_priority"], "none")

    def test_disagreement_always_requires_high_priority_review(self):
        results = compare_model_results(
            [row("a.png", prediction="Forest")],
            [row("a.png", prediction="River")],
        )
        result = results[0]
        self.assertEqual(result["comparison_status"], "disagreement")
        self.assertIs(result["agreement"], False)
        self.assertTrue(result["review_required"])
        self.assertEqual(result["review_priority"], "high")
        self.assertIn("model_disagreement", result["review_reasons"])

    def test_optional_low_score_and_margin_rules_are_model_specific(self):
        results = compare_model_results(
            [row("a.png", top1=0.55, top2=0.50)],
            [row("a.png", top1=0.95, top2=0.01)],
            low_score_threshold=0.60,
            margin_threshold=0.10,
        )
        result = results[0]
        self.assertEqual(result["comparison_status"], "agreement")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["review_priority"], "medium")
        self.assertIn("resnet50_low_top1_score", result["review_reasons"])
        self.assertIn("resnet50_small_top1_top2_margin", result["review_reasons"])
        self.assertNotIn("efficientnet_low_top1_score", result["review_reasons"])

    def test_partial_model_failure_is_reviewed(self):
        results = compare_model_results(
            [row("a.png")],
            [row("a.png", status="error", error="checkpoint unavailable")],
        )
        result = results[0]
        self.assertEqual(result["comparison_status"], "partial_failure")
        self.assertIsNone(result["agreement"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["review_priority"], "high")
        self.assertIn("efficientnet_inference_failed", result["review_reasons"])

    def test_both_model_failures_are_retained(self):
        results = compare_model_results(
            [row("bad.png", status="error")],
            [row("bad.png", status="error")],
        )
        self.assertEqual(results[0]["comparison_status"], "failed")
        self.assertTrue(results[0]["review_required"])
        self.assertIsNone(results[0]["agreement"])

    def test_filename_alignment_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "filename mismatch"):
            compare_model_results([row("a.png")], [row("b.png")])

    def test_result_lengths_must_match(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            compare_model_results([row("a.png")], [])

    def test_threshold_validation(self):
        for invalid in (-0.1, 1.1, float("nan"), "bad", True):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    compare_model_results([row("a.png")], [row("a.png")], low_score_threshold=invalid)

    def test_summary_uses_only_fully_compared_for_agreement_rate(self):
        rows = compare_model_results(
            [row("a.png"), row("b.png", prediction="Forest"), row("c.png")],
            [row("a.png"), row("b.png", prediction="River"), row("c.png", status="error")],
        )
        summary = summarize_comparison_results(rows)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["fully_compared"], 2)
        self.assertEqual(summary["agreement"], 1)
        self.assertEqual(summary["disagreement"], 1)
        self.assertEqual(summary["partial_failure"], 1)
        self.assertAlmostEqual(summary["agreement_rate"], 50.0)
        self.assertEqual(summary["review_required"], 2)

    def test_csv_exports_include_all_rows_and_review_only_rows(self):
        rows = compare_model_results(
            [row("a.png"), row("b.png")],
            [row("a.png"), row("b.png", prediction="River")],
        )
        all_csv = list(csv.DictReader(io.StringIO(comparison_results_to_csv(rows).decode("utf-8-sig"))))
        review_csv = list(csv.DictReader(io.StringIO(review_queue_to_csv(rows).decode("utf-8-sig"))))
        self.assertEqual(len(all_csv), 2)
        self.assertEqual(len(review_csv), 1)
        self.assertEqual(review_csv[0]["filename"], "b.png")

    def test_csv_formula_injection_protection(self):
        malicious = row('=HYPERLINK("https://bad.test","click")', prediction="Forest")
        results = compare_model_results([malicious], [malicious])
        full = list(csv.DictReader(io.StringIO(comparison_results_to_csv(results).decode("utf-8-sig"))))
        self.assertTrue(full[0]["filename"].startswith("'="))


if __name__ == "__main__":
    unittest.main()
