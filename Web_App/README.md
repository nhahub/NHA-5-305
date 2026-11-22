# Web Application — Streamlit

The main repository documentation is [../README.md](../README.md). This subfolder contains the **inference-only** web application.

Launch from repository root:

```bash
python -m pip install -r Web_App/requirements.txt
python Web_App/verify_package.py
streamlit run Web_App/app.py
```

For Streamlit Community Cloud, use **`Web_App/app.py`** as the entrypoint and **Python 3.11**.

The three trained model checkpoints are committed directly to the repository as regular Git binary files. **Git LFS is not required.**

Training data and the research notebook are **not required** to run the web application.

RGB input: JPG/PNG. 13-band multispectral input: authentic, compatible GeoTIFF with verified order/scaling. Predictions are limited to 10 EuroSAT classes; softmax scores are not calibrated probabilities.


## Batch classification and Smart Review

The Batch tab provides two RGB workflows:

1. **Single-model batch** — run either ResNet50 RGB or EfficientNet-B0 RGB.
2. **Multi-model comparison** — run both models on the same batch and compare their
   per-image Top-1 predictions.

The comparison dashboard reports agreement, disagreement, partial/total model
failures, and the number of images flagged for human review. Disagreements and
model failures are always flagged. Optional rules can flag low Top-1 scores or
small Top-1-vs-Top-2 margins independently for each model.

The app exports:
- `model_comparison_results.csv` — all per-image comparison results.
- `smart_review_queue.csv` — only images requiring human review.

Model execution is sequential to reduce peak memory use. The existing single-image
multispectral TIFF workflow is unchanged. See
[`BATCH_CLASSIFICATION_GUIDE.md`](BATCH_CLASSIFICATION_GUIDE.md) for limits,
interpretation, and operating instructions.

**Important:** agreement does not establish correctness. Softmax scores are
uncalibrated and optional thresholds are review heuristics, not guarantees.

Verification details: see [`../SMART_REVIEW_VERIFICATION.md`](../SMART_REVIEW_VERIFICATION.md).
