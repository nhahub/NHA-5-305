"""Pure reporting utilities; no trained models or Streamlit dependency."""
from __future__ import annotations

import csv
import io
from collections import Counter
from statistics import mean

from batch_inference import RESULT_COLUMNS


def summarize_results(rows: list[dict], classes: list[str] | tuple[str, ...]) -> dict:
    """Calculate proportions of classified *images*, never geographic area."""
    classes = list(classes)
    successful = [row for row in rows if row["status"] == "success"]
    failures = len(rows) - len(successful)
    counts = Counter(row["predicted_class"] for row in successful)
    distribution = [
        {"class": name, "count": counts[name], "percentage": (100.0 * counts[name] / len(successful)) if successful else 0.0}
        for name in classes
    ]
    scores = [float(row["top1_score"]) for row in successful]
    return {
        "total": len(rows), "successful": len(successful), "failed": failures,
        "distribution": distribution,
        "mean_top1_score": mean(scores) if scores else None,
        "min_top1_score": min(scores) if scores else None,
        "max_top1_score": max(scores) if scores else None,
    }


def _safe_csv_text(value: object) -> object:
    """Prevent uploaded filenames/errors being interpreted as spreadsheet formulas."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def results_to_csv(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _safe_csv_text(row.get(key, "")) for key in RESULT_COLUMNS})
    return stream.getvalue().encode("utf-8-sig")


def summary_to_csv(summary: dict) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["class", "count", "percentage"])
    writer.writeheader()
    writer.writerows(summary["distribution"])
    return stream.getvalue().encode("utf-8-sig")
