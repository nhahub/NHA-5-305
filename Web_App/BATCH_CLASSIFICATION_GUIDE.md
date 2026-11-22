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
├── app.py                       # updated: single-model and multi-model Batch modes
├── inference.py                 # existing: verified checkpoint loading and RGB preprocessing
├── batch_inference.py           # updated: safe batch execution for one or both RGB models
├── batch_analytics.py           # existing: single-model summary/CSV exports
├── smart_review.py              # new: agreement, review rules, summaries and comparison CSVs
├── BATCH_CLASSIFICATION_GUIDE.md
└── tests/
    ├── test_batch_inference.py
    ├── test_batch_analytics.py
    ├── test_smart_review.py
    └── test_inference.py
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

**Important:** Keep the existing `test_inference.py` coverage. It now checks the four-tab
layout and includes a simulated Multi-Model Comparison UI run. Do not delete or skip the
original checkpoint, RGB, or multispectral tests.

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

The implementation includes automated coverage for single-model batch inference, multi-model
execution, model disagreement, partial model failures, image/result alignment, review rules,
CSV exports, formula-injection protection, and Streamlit comparison-mode wiring. The
`SMART_REVIEW_VERIFICATION.md` report records the checks run against the supplied real
checkpoints. A live Streamlit browser session should still be launched locally before
committing or deploying the feature.

## Next increments

1. Run the complete test suite and package verifier in the full Windows repository.
2. Launch the app and test Multi-Model Comparison with the 100 real EuroSAT sample images.
3. Validate optional review thresholds on a held-out validation set before operational use.
4. Consider an independent geographic evaluation and optional multispectral batch support.


## Smart Review & Multi-Model Comparison

The Batch tab now offers a second workflow: **Multi-model comparison**.

### Workflow

1. Choose **Multi-model comparison**.
2. Upload multiple JPG/PNG files or one ZIP archive (the same input limits apply).
3. Choose the inference batch size.
4. Optionally enable low Top-1 score and/or small Top-1-vs-Top-2 margin review rules.
5. Run comparison. ResNet50 RGB and EfficientNet-B0 RGB process the same ordered inputs.
6. Inspect the comparison dashboard, filter by status or predicted class, and download reports.

### Comparison states

| State | Meaning | Review handling |
| --- | --- | --- |
| `agreement` | Both models succeeded and predicted the same class | Not automatically flagged unless an optional heuristic triggers |
| `disagreement` | Both models succeeded but predicted different classes | Always flagged, high priority |
| `partial_failure` | One model succeeded and the other failed | Always flagged, high priority |
| `failed` | Both model inferences failed for that image | Always flagged, high priority |

A filename/order check is applied before comparisons are assembled. A failed result from
one model does not discard a successful result from the other model.

### Optional review heuristics

- **Low Top-1 score:** flags a model result below the user-selected threshold.
- **Small Top-1-vs-Top-2 margin:** flags a model result when its own Top-1 score is too close
  to its own Top-2 score.

These checks are applied separately to each model. Scores from different architectures
are not compared to each other. Both optional rules are disabled by default. Softmax
scores are uncalibrated, so threshold values should be treated as triage heuristics and
validated on a suitable validation set before operational use.

### Downloads

- `model_comparison_results.csv` contains every uploaded filename, both model statuses,
  predictions, Top-1/Top-2/Top-3 details, margins, comparison state, and review reasons.
- `smart_review_queue.csv` contains only images with `review_required = True`.

CSV output protects user-controlled text values against spreadsheet formula injection.

### Resource behavior and scope

Both models are run **sequentially**, not concurrently, to reduce peak memory usage.
Each checkpoint is loaded once during its model pass. Uploaded image bytes are read once
and reused for both passes. Multi-model comparison is RGB-only; the existing individual
multispectral TIFF workflow is unchanged. Checkpoints are not modified and no training
is performed.

### Tests

The automated tests cover model invocation count, image/result alignment, disagreement,
partial failures, corrupt files, optional thresholds, summary metrics, CSV exports,
formula-injection protection, and Streamlit comparison-mode wiring.

Run all project tests and checkpoint verification from the repository root:

```powershell
python -m unittest discover -s Web_App/tests -v
python Web_App/verify_package.py
```

