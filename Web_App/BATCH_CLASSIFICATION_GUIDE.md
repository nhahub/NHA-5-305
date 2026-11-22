# Batch RGB Classification & Analytics — MVP

This is an **inference-only extension** of the project's existing Streamlit web application. It reuses the final ResNet50 RGB and EfficientNet-B0 RGB checkpoints and the original `preprocess_rgb` transformation. **No model retraining is needed.** The 13-band multispectral single-image workflow remains unchanged.

## Features

- Batch upload of up to **100 RGB satellite patches** as individual JPG/PNG images or one ZIP archive.
- CPU inference in configurable batches of **1, 2, 4, 8, or 16** images; default is 8.
- One result row per supplied file, including invalid images and any failures.
- Predicted land-cover class, Top-3 classes and uncalibrated softmax scores.
- Progress indicator and class distribution analytics, including all ten EuroSAT classes.
- Two downloadable CSV files: full results (including failed files) and class-count summary.
- Existing single-image RGB and 13-band TIFF interfaces retained.

## Placement

Copy the included files into the matching paths in the existing Git repository:

```text
Web_App/
├── app.py                       # updated: extra Batch tab
├── inference.py                 # updated: predict_tensors, same original preprocessing and weights
├── batch_inference.py           # new: validation, ZIP, batched model execution
├── batch_analytics.py           # new: summary/CSV exports
├── BATCH_CLASSIFICATION_GUIDE.md
└── tests/
    ├── test_batch_inference.py
    └── test_batch_analytics.py
```

The actual project must **retain** `Web_App/checkpoints/`, `Web_App/deployment_manifest.json`, `Web_App/requirements.txt`, and its pre-existing tests. No model binaries are included in this upgrade package.

## Running and testing on Windows (from repository root)

Use the project's already-installed Python 3.11 environment:

```powershell
cd D:\NHA-5-305
D:\Land-Type-Classification\.venv\Scripts\python.exe -m unittest discover -s Web_App/tests -v
D:\Land-Type-Classification\.venv\Scripts\python.exe Web_App\verify_package.py
D:\Land-Type-Classification\.venv\Scripts\python.exe -m streamlit run Web_App\app.py
```

**Important:** The existing `test_inference.py` may contain a simulated Streamlit UI assertion written for **three tabs**. The upgraded app now has **four tabs**, so update any hard-coded UI count or uploader expectations in that existing test. Do not delete or skip the original tests; inspect any failures and adapt the simulation to the new interface.

## Safe limits (MVP)

- 100 files per batch, 10 MiB per image, 100 MiB total input data.
- ZIPs at most 50 MiB compressed and 100 MiB declared uncompressed.
- ZIP processing is in memory; the app never extracts an archive into the local filesystem.
- Archive path traversal, encrypted entries and extreme compression ratios are rejected.
- Input must be real JPEG or PNG image content; extension-only checks are not trusted.
- Maximum 16 million pixels per image.

These limits are **application-side checks**. The browser/Streamlit upload mechanism may allocate memory before validation. Measure memory use before increasing limits, especially on Community Cloud.

## Data interpretation

The dashboard displays **percentages of successfully classified image patches**, not percentages of real geographic land area. Softmax scores are **not calibrated probabilities** and should not be treated as guarantees of correctness. The trained models recognize the ten EuroSAT classes; `Desert` is not a class.

## Testing status

The new code has **18 unit tests that passed with a deterministic mock PyTorch model** in the artifact preparation environment. This checks batch-versus-single prediction consistency, malformed inputs, ZIP safeguards, summary math and CSV export. **Real-checkpoint integration tests and live Streamlit browser testing still need to be run on the user's Windows machine**; the real checkpoints and Streamlit runtime are not included here.

## Next increments

1. Validate the real-checkpoint RGB results on 20 EuroSAT images, compare batch versus single image predictions.
2. Benchmark 100 images (time, throughput, memory), then evaluate larger input sizes in a suitable runtime.
3. Add confidence-calibration-based human review and optional multispectral batch mode.
4. Update existing README and UI wiring tests, then commit to `feature/batch-analytics` after verification.
