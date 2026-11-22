"""Pure multi-model comparison and human-review reporting utilities.

Agreement is a consistency signal, not evidence that a prediction is correct.
Softmax scores are uncalibrated and thresholds are optional heuristics.
"""
from __future__ import annotations

import csv
import io
import math
from collections import Counter
from typing import Any, Sequence

COMPARISON_COLUMNS = (
    "filename",
    "resnet50_status", "resnet50_prediction", "resnet50_top1_score",
    "resnet50_top2_class", "resnet50_top2_score", "resnet50_top3_class",
    "resnet50_top3_score", "resnet50_margin", "resnet50_error",
    "efficientnet_status", "efficientnet_prediction", "efficientnet_top1_score",
    "efficientnet_top2_class", "efficientnet_top2_score", "efficientnet_top3_class",
    "efficientnet_top3_score", "efficientnet_margin", "efficientnet_error",
    "comparison_status", "agreement", "review_required", "review_priority",
    "review_reasons",
)
REVIEW_COLUMNS = (
    "filename", "resnet50_prediction", "resnet50_top1_score",
    "efficientnet_prediction", "efficientnet_top1_score",
    "comparison_status", "review_priority", "review_reasons",
    "resnet50_error", "efficientnet_error",
)


def _optional_threshold(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number between 0 and 1.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1.") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1.")
    return number


def validate_review_thresholds(
    low_score_threshold: float | None,
    margin_threshold: float | None,
) -> tuple[float | None, float | None]:
    """Validate optional heuristic thresholds before running expensive inference."""
    return (
        _optional_threshold(low_score_threshold, "Low-score threshold"),
        _optional_threshold(margin_threshold, "Margin threshold"),
    )


def _score(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _model_fields(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    top1 = _score(row, "top1_score")
    top2 = _score(row, "top2_score")
    margin = max(0.0, top1 - top2) if top1 is not None and top2 is not None else None
    return {
        f"{prefix}_status": row.get("status", "error"),
        f"{prefix}_prediction": row.get("predicted_class", ""),
        f"{prefix}_top1_score": top1,
        f"{prefix}_top2_class": row.get("top2_class", ""),
        f"{prefix}_top2_score": top2,
        f"{prefix}_top3_class": row.get("top3_class", ""),
        f"{prefix}_top3_score": _score(row, "top3_score"),
        f"{prefix}_margin": margin,
        f"{prefix}_error": row.get("error", ""),
    }


def compare_model_results(
    resnet_rows: Sequence[dict[str, Any]],
    efficientnet_rows: Sequence[dict[str, Any]],
    *,
    low_score_threshold: float | None = None,
    margin_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Compare ordered per-image outputs from ResNet50 and EfficientNet-B0."""
    low_score_threshold, margin_threshold = validate_review_thresholds(
        low_score_threshold, margin_threshold
    )
    if len(resnet_rows) != len(efficientnet_rows):
        raise ValueError("Model result lists must contain the same number of images.")

    results: list[dict[str, Any]] = []
    for index, (resnet, efficient) in enumerate(zip(resnet_rows, efficientnet_rows)):
        resnet_name = str(resnet.get("filename", ""))
        efficient_name = str(efficient.get("filename", ""))
        if resnet_name != efficient_name:
            raise ValueError(
                f"Model result filename mismatch at row {index}: "
                f"{resnet_name!r} != {efficient_name!r}"
            )

        r_ok = resnet.get("status") == "success"
        e_ok = efficient.get("status") == "success"
        reasons: list[str] = []

        if r_ok and e_ok:
            agreement: bool | None = (
                str(resnet.get("predicted_class", "")) ==
                str(efficient.get("predicted_class", ""))
            )
            comparison_status = "agreement" if agreement else "disagreement"
            if not agreement:
                reasons.append("model_disagreement")
        elif r_ok or e_ok:
            agreement = None
            comparison_status = "partial_failure"
            if not r_ok:
                reasons.append("resnet50_inference_failed")
            if not e_ok:
                reasons.append("efficientnet_inference_failed")
        else:
            agreement = None
            comparison_status = "failed"
            reasons.extend(("resnet50_inference_failed", "efficientnet_inference_failed"))

        model_pairs = (
            ("resnet50", resnet, r_ok),
            ("efficientnet", efficient, e_ok),
        )
        if r_ok and e_ok:
            for prefix, row, successful in model_pairs:
                if not successful:
                    continue
                top1 = _score(row, "top1_score")
                top2 = _score(row, "top2_score")
                if low_score_threshold is not None and top1 is not None and top1 < low_score_threshold:
                    reasons.append(f"{prefix}_low_top1_score")
                margin = (top1 - top2) if top1 is not None and top2 is not None else None
                if margin_threshold is not None and margin is not None and margin < margin_threshold:
                    reasons.append(f"{prefix}_small_top1_top2_margin")

        review_required = bool(reasons)
        if comparison_status in ("disagreement", "partial_failure", "failed"):
            priority = "high"
        elif review_required:
            priority = "medium"
        else:
            priority = "none"

        result = {
            "filename": resnet_name,
            **_model_fields(resnet, "resnet50"),
            **_model_fields(efficient, "efficientnet"),
            "comparison_status": comparison_status,
            "agreement": agreement,
            "review_required": review_required,
            "review_priority": priority,
            "review_reasons": ";".join(reasons),
        }
        results.append(result)
    return results


def summarize_comparison_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize comparison outcomes; agreement rate excludes failed comparisons."""
    statuses = Counter(row.get("comparison_status", "failed") for row in rows)
    fully_compared = statuses["agreement"] + statuses["disagreement"]
    agreements = statuses["agreement"]
    review_count = sum(bool(row.get("review_required")) for row in rows)
    return {
        "total": len(rows),
        "fully_compared": fully_compared,
        "agreement": agreements,
        "disagreement": statuses["disagreement"],
        "partial_failure": statuses["partial_failure"],
        "failed": statuses["failed"],
        "review_required": review_count,
        "agreement_rate": (100.0 * agreements / fully_compared) if fully_compared else None,
        "status_distribution": [
            {"status": name, "count": statuses[name]}
            for name in ("agreement", "disagreement", "partial_failure", "failed")
        ],
    }


def _safe_csv_text(value: Any) -> Any:
    """Prevent user-controlled filenames/errors from becoming spreadsheet formulas."""
    if isinstance(value, str):
        if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
            return "'" + value
    return value


def _to_csv(rows: Sequence[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _safe_csv_text(row.get(key, "")) for key in columns})
    return stream.getvalue().encode("utf-8-sig")


def comparison_results_to_csv(rows: Sequence[dict[str, Any]]) -> bytes:
    """Export every comparison row, including partial and total failures."""
    return _to_csv(rows, COMPARISON_COLUMNS)


def review_queue_to_csv(rows: Sequence[dict[str, Any]]) -> bytes:
    """Export only rows flagged for human review."""
    selected = [row for row in rows if bool(row.get("review_required"))]
    return _to_csv(selected, REVIEW_COLUMNS)
