# Verification Report — Streamlit Community Cloud package

**Date prepared:** 2026-11-21  
**Model run:** `run_20261121_003103_UTC`  
**Source:** Executed final notebook and original `.pth`, `.json`, and `.csv` artifacts provided for the project.

## Completed checks

| Check | Result | Evidence / scope |
| --- | --- | --- |
| Actual ResNet50 RGB checkpoint | **PASS** | SHA256 verified and loaded with `torch.load(weights_only=True)` |
| Actual EfficientNet-B0 RGB checkpoint | **PASS** | SHA256 verified and loaded |
| Actual ResNet50 13-band checkpoint | **PASS** | SHA256 verified and loaded |
| RGB inference | **PASS** | Both real model files produced top-3 predictions on synthetic 64×64 RGB test patches |
| Multispectral inference | **PASS** | Real model produced a prediction on synthetic 13-band GeoTIFF inputs |
| Multispectral band ordering | **PASS** | Shuffled TIFF with per-band descriptions returned equivalent preprocessed tensor to original ordering |
| Invalid TIFF handling | **PASS** | 3-band TIFF rejected and unlabeled order requires explicit confirmation |
| Git LFS metadata | **PASS** | `.gitattributes` tracks `Web_App/checkpoints/*.pth`; files excluded from neither `.gitignore` nor package |
| Streamlit view wiring | **PASS (simulated)** | Mocked Streamlit module checked that 3 tabs, two uploaders, and two classify actions are configured |
| Application Python syntax | **PASS** | `py_compile` |
| Automated unit/integration tests | **PASS** | 6 tests run against real checkpoint files and simulated Streamlit API |
| Notebook final audit | **PASS** | Supplied `final_run_audit.json` says evaluation PASSED, checkpoint hashes verified, six notebook-side smoke tests |

### Notes on test inputs

Predictions used **synthetic/random inputs** to test file handling, architecture integrity, and inference. They **are not extra accuracy measurements**, and are not substituted for reported EuroSAT results.

### Limitations / not yet verified

1. The **real Streamlit server could not be started in the artifact-building environment** because the Streamlit package was not installed and external package installation was blocked by network access. The frontend was syntax-checked and exercised using simulated Streamlit UI primitives, but this is not equivalent to live browser verification.
2. **Streamlit Community Cloud deployment is not done**; only the account owner can authorize access to their GitHub repository and deploy. No live public URL has been checked.
3. Git LFS upload/checkout from the user's GitHub account and **cloud resource limits** have not been tested. The package contains full checkpoint binaries; cloud capacity must be confirmed after deployment.
4. Benchmarks are from the **provided final EuroSAT execution** (4,050 holdout samples per model), not measured from synthetic smoke inputs and not an independent geographic/real-world validation.

## Final benchmark results (source manifest / CSV)

| Model | Accuracy | Macro F1 | Test mistakes |
| --- | ---: | ---: | ---: |
| ResNet50 RGB | 97.90% | 97.86% | 85 |
| EfficientNet-B0 RGB | 97.04% | 96.93% | 120 |
| ResNet50 Multispectral | 98.69% | 98.63% | 53 |

## Commands for user-side final verification

```bash
python Web_App/verify_package.py
python -m unittest discover -s Web_App/tests -v
streamlit run Web_App/app.py
```

On Streamlit Community Cloud, choose **Python 3.11** and **`Web_App/app.py`** as the main file. Confirm all 3 model pathways from the live public browser page before asserting the cloud deployment is complete.
