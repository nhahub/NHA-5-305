# 🛰️ Land Type Classification Using Deep Learning and Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-186e63)](https://github.com/phelber/EuroSAT)

> **Graduation Project** — End-to-end land-cover classification from satellite imagery: dataset exploration, transfer learning, reproducible evaluation, interpretability, verified checkpoints, and a deployable Streamlit web application.

**Live demo:** _Add the actual `https://...streamlit.app` URL here after deployment and browser verification._

## Table of contents

1. [Project overview](#project-overview)
2. [Problem statement and objectives](#problem-statement-and-objectives)
3. [Dataset and land-cover classes](#dataset-and-land-cover-classes)
4. [Methods and architectures](#methods-and-architectures)
5. [Data preparation and reproducibility](#data-preparation-and-reproducibility)
6. [Training and evaluation methodology](#training-and-evaluation-methodology)
7. [Final experimental results](#final-experimental-results)
8. [Interpretability and error analysis](#interpretability-and-error-analysis)
9. [Repository organization](#repository-organization)
10. [Running the web application](#running-the-web-application)
11. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
12. [Checkpoint integrity and Git LFS](#checkpoint-integrity-and-git-lfs)
13. [Reproducing the notebook](#reproducing-the-notebook)
14. [Input specifications and limitations](#input-specifications-and-limitations)
15. [Troubleshooting](#troubleshooting)
16. [References and acknowledgements](#references-and-acknowledgements)

## Project overview

This project classifies remotely sensed land-cover patches into **10 EuroSAT categories** using pretrained convolutional neural networks. It compares **ResNet50 and EfficientNet-B0** on three-channel RGB imagery and extends **ResNet50 to all 13 Sentinel-2 spectral channels**.

The project has two independent parts:

- **Research notebook (`Land_Type_Classification_Final.ipynb`)** — retrieves data, performs exploratory data analysis (EDA), constructs reproducible splits, fine-tunes networks, evaluates held-out samples, analyzes errors, generates Grad-CAM visualizations, and exports validated checkpoints.
- **Deployment app (`Web_App/`)** — loads the *already-trained* checkpoints on CPU and classifies user-uploaded satellite images. **The app does not require EuroSAT, Google Drive, Colab, or retraining.**

### End-to-end workflow

```text
EuroSAT RGB JPEGs + 13-band GeoTIFFs
                |
     EDA and data quality checks
                |
     Stratified split (seed 42)
                |
       Training-only augmentation
                |
  +-------------+------------+----------------+
  |                          |                |
ResNet50 RGB           EfficientNet-B0    ResNet50 13-band
  |                          |                |
  +-------------+------------+----------------+
                |
   Validation-selected best checkpoints
                |
       Held-out test evaluation
                |
 Accuracy • Macro-F1 • Confusion matrices
 Grad-CAM • Prediction examples • Error audit
                |
   .pth weights + deployment_manifest.json
                |
     Streamlit web app on CPU
                |
 Upload image → preprocess → predict → Top-3
```

## Problem statement and objectives

Satellite imagery contains visual and spectral patterns useful for land-cover recognition. Manually interpreting large volumes of imagery can be slow. This project investigates whether transfer learning can classify small satellite patches and whether additional spectral bands improve classification on the EuroSAT benchmark.

**Objectives:**

- Build a reproducible **10-class** supervised classification pipeline.
- Compare two RGB transfer-learning architectures on **identical held-out RGB test samples**.
- Train a 13-channel multispectral model on the **paired** EuroSAT patches.
- Evaluate accuracy, macro precision, macro recall, macro F1, and class-level confusion.
- Explain selected RGB and multispectral decisions with Grad-CAM.
- Export self-contained, verifiable model weights for standalone inference.
- Make predictions accessible through a simple web application.

## Dataset and land-cover classes

**EuroSAT** contains **27,000** labeled satellite image patches across 10 classes. The RGB version contains 64×64 JPEG images; the multispectral version includes **13-band Sentinel-2 GeoTIFFs**. Multispectral acquisition uses the project's documented official archive, `EuroSAT_MS.zip`.

| Split | Images | Fraction |
| --- | ---: | ---: |
| Training | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Held-out test | 4,050 | 15% |
| **Total** | **27,000** | **100%** |

**Class order saved in every checkpoint:**

| Index | Class | Description |
| ---: | --- | --- |
| 0 | `AnnualCrop` | Annual agricultural crops |
| 1 | `Forest` | Forested land |
| 2 | `HerbaceousVegetation` | Herbaceous vegetation |
| 3 | `Highway` | Major roads and highways |
| 4 | `Industrial` | Industrial facilities |
| 5 | `Pasture` | Pasture and grazing land |
| 6 | `PermanentCrop` | Permanent crops |
| 7 | `Residential` | Residential development |
| 8 | `River` | Rivers |
| 9 | `SeaLake` | Seas and lakes |

**Important:** `Desert` is **not** a trained category. Do not claim reliable desert classification without extending the dataset, retraining and evaluating an additional class.

The dataset's original images and archives are **not** committed to this repository. The notebook obtains the original data through the published source and `torchvision.datasets.EuroSAT`.

## Methods and architectures

| Experiment | Architecture | Channels | Input to network | Fine-tuned components |
| --- | --- | ---: | --- | --- |
| RGB baseline | ImageNet-pretrained **ResNet50** | 3 | 224×224 | `layer4`, final `fc` |
| RGB comparison | ImageNet-pretrained **EfficientNet-B0** | 3 | 224×224 | `features[-2]`, `features[-1]`, classifier |
| Multispectral extension | Modified ImageNet-pretrained **ResNet50** | 13 | 224×224 | 13-channel stem (`conv1`), `layer4`, final `fc` |

Each architecture replaces the final classification head with a 10-class output layer. Multispectral ResNet50 initializes its 13-channel stem from the mean of the pretrained RGB kernels, repeated and rescaled as described in the notebook.

**The model architectures are not identical in optimization settings.** The RGB experiments use Adam with learning rate `1e-4`, while the multispectral experiment uses AdamW (`3e-4`, weight decay `1e-4`) and a validation-loss scheduler. The multispectral comparison is therefore a useful observed benchmark comparison, **not a strictly controlled proof** that spectral channels alone explain the improvement.

## Data preparation and reproducibility

### Split policy

- Seed: **42**; separate stratified training/validation/test membership.
- RGB and multispectral files are checked by **class and filename stem** to establish paired examples. The final audit reports paired RGB/multispectral test membership.
- Test data is excluded from training and validation-based checkpoint selection.
- No image index belongs to two different splits.
- Splitting is **image-level**, **not geographically independent**; neighboring patches or acquisition conditions may be correlated.

### RGB preprocessing

- Input source: original 64×64 EuroSAT RGB image.
- Train-only augmentation: random horizontal/vertical flips and ±15° rotations.
- Resize to **224×224** using the deterministic evaluation transformation.
- Convert to tensor and normalize with ImageNet statistics: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- Validation, test, and website inference use **no random augmentation**.

**Resolution caveat:** Upscaling 64×64 images to 224×224 does not create new physical detail; it changes the model input representation.

### Multispectral preprocessing

- Uses Rasterio and **13 spectral bands**, in the exact order:

  ```text
  B01, B02, B03, B04, B05, B06, B07, B08,
  B09, B10, B11, B12, B8A
  ```

- Handles invalid/nodata pixels as described in the notebook.
- Computes **per-band mean and standard deviation using training images only**, then applies those saved values to validation/test/deployment inputs.
- Resizes 64×64 normalized tensors to 224×224 using bilinear interpolation (`align_corners=False`).
- Applies flips and right-angle rotations **only during training**.
- Does **not** estimate new normalization statistics from website uploads.

### Reproducibility protections

- `random`, NumPy and PyTorch seeds.
- Deterministic CuDNN settings where applicable.
- Worker-seeding and dedicated data-loader generators.
- Validation-loss selection of the best checkpoint, with early stopping.
- Independent CPU loading of saved models, six notebook-side deployment smoke checks, and SHA256 checkpoint verification.
- Machine-readable output: `deployment_manifest.json`, `final_run_audit.json`, and `final_test_metrics.csv`.

The saved run audit reports `evaluation: PASSED`, `checkpoint_hashes_verified: true`, and `smoke_test_count: 6`. See the complete notebook for the original execution logs and figures; these files are not a substitute for inspecting the saved notebook's outputs.

## Training and evaluation methodology

**Training:** supervised multiclass cross-entropy. The checkpoints are selected using **validation loss**, not test-set accuracy. The RGB baseline and EfficientNet use transfer learning and validation-based early stopping. The multispectral model also uses a `ReduceLROnPlateau` scheduler.

**Evaluation:** one fixed held-out set of 4,050 samples for each model. The notebook reports:

- Test accuracy.
- Macro precision, macro recall, and macro F1.
- Per-class precision, recall and F1 via a classification report.
- Confusion matrices and analysis of commonly confused land-cover categories.
- Correct/incorrect prediction examples and confidence scores.
- Training duration measured in the final execution, not copied from previous runs.

**Accuracy** is correct predictions / test images. **Macro F1** gives every class equal weight when averaging per-class F1 scores.

## Final experimental results

**Source of truth:** `Web_App/final_test_metrics.csv` and `Web_App/deployment_manifest.json` from `run_20261121_003103_UTC`.

| Model | Test samples | Accuracy | Macro precision | Macro recall | Macro F1 | Errors | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50 RGB | 4,050 | **97.90%** | 97.85% | 97.89% | 97.86% | 85 | 4 |
| EfficientNet-B0 RGB | 4,050 | **97.04%** | 96.90% | 96.98% | 96.93% | 120 | 14 |
| ResNet50 Multispectral (13 bands) | 4,050 | **98.69%** | 98.63% | 98.64% | 98.63% | 53 | 20 |

The multispectral ResNet50 achieved the highest observed EuroSAT test accuracy in the final run. This **does not** establish its accuracy on new Egyptian imagery or other geographic regions.

**Final-run measured training time:** ResNet50 RGB ~746.5 s, EfficientNet-B0 ~956.0 s, multispectral ResNet50 ~5017.3 s (environment-dependent).

Previous notebook versions reported slightly different results; the table above refers **only to the final validated run**. The optional native-64×64 ablation has been removed from the final notebook; the main RGB input pipeline continues to resize 64×64 to 224×224.

## Interpretability and error analysis

The notebook includes:

- Class-wise confusion matrices and reports for the models.
- Examples of correctly and incorrectly classified images.
- Grad-CAM visualization for RGB and 13-band ResNet50 to examine influential image regions.
- Joint RGB versus multispectral error analysis on paired held-out samples.

Grad-CAM helps visualize regions relevant to a prediction but is **not** a proof of causality or geographic robustness. Full visual outputs are preserved in the final `.ipynb`.

## Repository organization

```text
Land-Type-Classification/
├── Land_Type_Classification_Final.ipynb   # Training, plots, metrics and saved outputs
├── README.md                              # Full project documentation (this file)
├── .gitattributes                         # Git LFS tracking for .pth checkpoints
├── .gitignore
├── .streamlit/
│   └── config.toml                        # Theme and upload limits
└── Web_App/
    ├── app.py                             # Streamlit user interface
    ├── inference.py                       # PyTorch / TIFF preprocessing and prediction
    ├── requirements.txt                   # CPU-oriented deployment dependencies
    ├── deployment_manifest.json           # File hashes and model metadata
    ├── final_test_metrics.csv             # Final benchmark numbers
    ├── final_run_audit.json               # Validation summary
    ├── verify_package.py                  # Real checkpoint smoke tests
    ├── tests/
    │   └── test_inference.py              # Unit and integration tests
    └── checkpoints/
        ├── best_resnet50_eurosat.pth
        ├── best_efficientnet_b0_eurosat_two_blocks.pth
        └── best_resnet50_eurosat_multispectral.pth
```

**Training data is excluded**. Model files are stored in Git LFS, preserving their exact filenames and bytes. The online app requires only `Web_App/` and the repo's `.streamlit/config.toml`; it does **not** open the notebook.

## Running the web application

### Prerequisites

- Python **3.11**, 64-bit (recommended for reproducible CPU installation).
- Internet on first setup to install dependencies.
- The **three real `.pth` files** in `Web_App/checkpoints/` (not Git LFS pointer files).

From the **repository root**:

```bash
python -m pip install -r Web_App/requirements.txt
python Web_App/verify_package.py
python -m unittest discover -s Web_App/tests -v
streamlit run Web_App/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) in Chrome. You can select **ResNet50 RGB** or **EfficientNet-B0 RGB** under the first tab or upload a properly formatted 13-band TIFF in the second tab. Each prediction shows the winning class, a softmax score, and the Top 3 scores.

The app runs **on CPU**, lazily loading only the selected model and releasing duplicate weight tensors to reduce memory usage on the free cloud tier. A GPU is not required for inference.

### Testing a real image

1. Obtain a **EuroSAT RGB JPG/PNG** patch you are licensed to use.
2. Open the **RGB Image** tab.
3. Upload the image, choose the RGB model, and click **Classify Image**.
4. Examine the predicted class and Top 3 scores.
5. For the multispectral model, use an authentic **13-band EuroSAT-compatible GeoTIFF**, confirm its ordering/scale, then classify under the second tab.

Do not use an ordinary smartphone photo to validate EuroSAT classification accuracy.

## Deploying to Streamlit Community Cloud

**Community Cloud** runs this app from **your GitHub repository**; no Docker, Hugging Face Spaces, Colab, or continuously running personal computer is necessary. Public deployment remains to be verified on your actual account.

### GitHub preparation

1. Install Git and **Git LFS** and run `git lfs install`.
2. Confirm the root `.gitattributes` tracks `Web_App/checkpoints/*.pth`.
3. Stage and commit this project. Check `git lfs ls-files` lists the **three** checkpoints before pushing.
4. Push to the GitHub repository/branch you own. Git LFS has a free allowance, subject to GitHub quotas.

> **Do not use GitHub's web upload for the `.pth` files.** The ~90 MiB ResNet50 weights exceed browser upload limits, and GitHub advises LFS for files above 50 MiB. Use Git with LFS.

### Cloud deployment form

1. Open **https://share.streamlit.io/** and sign in with GitHub.
2. Choose **Create app → Yup, I have an app**.
3. Select your GitHub repository and your branch (e.g. `main`).
4. Set **Main file path** to **`Web_App/app.py`**.
5. In **Advanced settings**, select **Python 3.11**.
6. Create/deploy the app and inspect its build logs.
7. Open the resulting `*.streamlit.app` URL from another browser/device and run separate RGB and multispectral tests.

Streamlit discovers `Web_App/requirements.txt` beside the selected entrypoint. The configuration at `.streamlit/config.toml` is at the repository root, as required by Community Cloud.

**Deployment status:** The app's *model inference code* has been tested locally with the actual checkpoints and a simulated interface test; **a live Streamlit web server and public Community Cloud deployment have not yet been verified** in the artifact-building environment. See `STREAMLIT_DEPLOYMENT_GUIDE_AR.md` and `VERIFICATION_REPORT.md` for clear next steps.

## Checkpoint integrity and Git LFS

Every `.pth` file was generated by the completed notebook and stores:

- `model_state_dict` (trained weights)
- Architecture and input channel count
- Class names and class order
- Expected network image size
- Preprocessing parameters: RGB ImageNet mean/std or multispectral per-band training mean/std, plus 13-band order

`deployment_manifest.json` contains SHA256 checksums of all three files. `Web_App/inference.py` checks the exact digest **before loading** each model. A changed, truncated, or invalid checkpoint produces an error rather than silently producing predictions.

This project uses **Git LFS** for `.pth` files. If you see a tiny text file beginning with `version https://git-lfs.github.com/spec/v1`, you have an LFS **pointer**, not the actual checkpoint. Install LFS and pull the real weights using `git lfs pull`.

**Never commit** personal credentials, Google Drive tokens, `secrets.toml`, or private files. Keep the deployment app public only if you intend to share the models and code publicly.

## Reproducing the notebook

The notebook is a **research/training artifact**, not a prerequisite for running the deployed application.

1. Open `Land_Type_Classification_Final.ipynb` in **Google Colab**.
2. Select a GPU runtime if retraining is needed; enable/authorize Google Drive backup.
3. Use **Runtime → Restart session and run all** for a clean execution.
4. Let the notebook download EuroSAT RGB and multispectral data, train the three networks, generate plots, and export checkpoints.
5. Check final `PASSED` audit and `final_run_audit.json`; separately review the notebook execution counts and absence of exceptions.
6. Copy the new `*.pth`, `deployment_manifest.json`, and metrics into `Web_App/` **as one consistent run**; do not mix checkpoints and manifest files from different runs.
7. Run `verify_package.py` and automated tests before committing updated checkpoints.

**Do not rerun training just to deploy the website.** The verified, supplied checkpoints are sufficient. GPU training takes considerable time, whereas CPU prediction runs from saved model files.

## Input specifications and limitations

| Property | RGB models | Multispectral model |
| --- | --- | --- |
| File format | JPG/JPEG, PNG | TIFF/GeoTIFF (`.tif`, `.tiff`) |
| Channels | 3 | Exactly 13 |
| Normalization | Checkpoint ImageNet mean/std | Training-only mean/std stored inside checkpoint |
| Resolution | Resized to 224×224 | Resampled to 224×224 |
| Expected source | EuroSAT-style RGB satellite patch | EuroSAT-compatible original Sentinel-2 13-band TIFF |
| Model output | 1 of 10 EuroSAT classes + Top 3 | Same 10 EuroSAT classes + Top 3 |

Additional qualifications:

- **No verified geographic transfer:** The held-out EuroSAT split does not prove generalization to independent Egyptian scenes or different countries.
- **Domain shift:** Spatial resolution, season, sensor, atmospheric correction and channel scaling can change predictions.
- **Missing `B10`:** Many Sentinel-2 Level-2A products lack band B10. Do not invent missing channels to fit the multispectral model.
- **Unknown class:** The model always chooses among the ten known categories, even for unsupported terrain like desert.
- **Uncalibrated scores:** A 99% softmax output is not a statistical guarantee of being correct.
- **No segmentation:** This is **patch classification**, not per-pixel semantic segmentation or mapping a large geographic area.
- **No new Egyptian benchmark:** A future optional Egypt evaluation would need independently checked labels and matched inputs; it is **not** part of the reported EuroSAT accuracy.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: streamlit` | Install `Web_App/requirements.txt` in Python 3.11. |
| `Checkpoint not found` | Confirm the three `.pth` files are under `Web_App/checkpoints/`. |
| `SHA256 mismatch` | Verify that Git LFS pulled the full model weights and the manifest matches that run. |
| Error about 13 bands | Use a TIFF with exactly 13 spectral channels, not ordinary RGB or 12-channel Level-2A. |
| Error about band order | Check TIFF band descriptions; for unnamed TIFFs, manually verify exact order and sensor value units. |
| GitHub blocks large file | Install Git LFS **before staging/committing** model files. |
| Streamlit Cloud out-of-memory | The app caches at most one loaded model, but free resources vary; review Cloud logs and resource limits. |
| App does not run after deploy | Confirm entrypoint `Web_App/app.py`, Python 3.11, LFS files, and requirements in `Web_App/`. |


## Smart Review & Multi-Model Comparison

The RGB batch workflow also supports **Multi-Model Comparison**. It runs the existing
ResNet50 RGB and EfficientNet-B0 RGB checkpoints on the same uploaded images, then
compares each pair of predictions by filename. This is an inference-only feature:
no retraining, fine-tuning, or checkpoint replacement is required.

### What it adds

- Run both trained RGB models against the same batch of JPG/PNG files or one ZIP archive.
- Load the models sequentially (one model at a time) to reduce peak memory use.
- Compare Top-1 predictions for every image and report agreement, disagreement,
  partial model failure, or total failure.
- Always place disagreements and model failures in the human-review queue.
- Optionally flag low Top-1 scores and small Top-1-vs-Top-2 margins using separate
  per-model thresholds. These optional rules are disabled unless selected.
- Filter comparison results and export all comparisons or only the review queue.

### Interpretation and limitations

- **Agreement is not proof of correctness.** Both models can make the same wrong prediction.
- The agreement rate is a consistency measure among images successfully processed
  by both models; it is not model accuracy.
- Softmax scores are uncalibrated. Optional score and margin thresholds are
  review heuristics, not calibrated confidence estimates, and scores are not
  compared directly across the two model architectures.
- Multi-model comparison currently supports RGB inputs only. The existing
  single-image 13-band multispectral workflow remains unchanged.
- A model failure is retained as a partial/failed comparison row and flagged for review.
- The comparison workflow does not modify checkpoints or train a new model.

### Run tests

```bash
python -m unittest discover -s Web_App/tests -v
python Web_App/verify_package.py
```

See [`SMART_REVIEW_VERIFICATION.md`](SMART_REVIEW_VERIFICATION.md) for the implementation
scope, test results, and verification limitations.


## References and acknowledgements

- Helber, P. et al., **EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification** — [paper](https://arxiv.org/abs/1709.00029), [dataset repository](https://github.com/phelber/EuroSAT).
- EuroSAT multispectral archive: [Zenodo](https://zenodo.org/records/7711810).
- [Sentinel-2 mission](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2).
- [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/), [Rasterio](https://rasterio.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/), [Streamlit](https://streamlit.io/).
- [Streamlit Community Cloud deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
- [GitHub Git LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

The experimental findings and deployment files are derived from the **actual final run and checkpoint artifacts supplied with this project**. Metrics are benchmark observations, not guarantees of real-world performance.
