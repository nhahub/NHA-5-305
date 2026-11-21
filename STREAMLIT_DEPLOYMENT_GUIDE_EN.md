# Deployment Guide — Land Type Classification (GitHub + Streamlit Community Cloud)

**Project:** DEPI graduation project, Land Type Classification Using Deep Learning and Satellite Imagery  
**GitHub repository:** https://github.com/nhahub/NHA-5-305  
**Prepared branch:** `feature/final-project`  
**Main application:** `Web_App/app.py`  
**Recommended Python:** 3.11  
**Checkpoint storage:** Regular Git binary files (**no Git LFS required**)  
**Status:** Local verification passed. GitHub publication, pull request merge and public deployment are pending until completed and checked on the owner's accounts.

This guide is written for the actual project structure and Git history. It does not require Docker, Hugging Face Spaces, Google Colab, retraining or a running personal computer after deployment.

## 1. Verify the repository layout

```text
NHA-5-305/
├── Land_Type_Classification_Final.ipynb
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── STREAMLIT_DEPLOYMENT_GUIDE_EN.md
├── VERIFICATION_REPORT.md
└── Web_App/
    ├── app.py
    ├── inference.py
    ├── requirements.txt
    ├── README.md
    ├── deployment_manifest.json
    ├── final_test_metrics.csv
    ├── final_run_audit.json
    ├── verify_package.py
    ├── tests/
    │   └── test_inference.py
    └── checkpoints/
        ├── best_resnet50_eurosat.pth
        ├── best_efficientnet_b0_eurosat_two_blocks.pth
        └── best_resnet50_eurosat_multispectral.pth
```

The original EuroSAT images and dataset archives are **not** part of the deployment repository. Streamlit loads the trained weights directly; it does not run the research notebook or train new models.

## 2. Model checkpoints and GitHub file limits

The three checkpoint files are committed as regular Git blobs, **not Git LFS pointers**:

| Checkpoint | Size (bytes) | Approximate size |
| --- | ---: | ---: |
| `best_efficientnet_b0_eurosat_two_blocks.pth` | 16,374,631 | 15.62 MiB |
| `best_resnet50_eurosat.pth` | 94,428,715 | 90.05 MiB |
| `best_resnet50_eurosat_multispectral.pth` | 94,640,191 | 90.26 MiB |

Total checkpoint storage is about **196 MiB**. GitHub warns about files larger than 50 MiB and blocks files larger than 100 MiB. The three current checkpoints are individually below the hard limit, so they can be pushed with regular Git. Binary weights make the repository larger; future changes to the weights should be planned carefully.

**Why not LFS?** An earlier attempt to upload these files via Git LFS failed because the repository's LFS budget was exceeded. The final feature branch was recreated to store the weights as normal Git files. **Do not reintroduce `.gitattributes` or LFS-tracking rules unless the team intentionally migrates storage and has an available LFS budget.**

Use command-line `git push`, not GitHub's browser file uploader, for these large binaries.

## 3. Verify the application locally (Windows PowerShell)

From the existing cloned repository:

```powershell
cd D:\NHA-5-305
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Web_App\requirements.txt
.\.venv\Scripts\python.exe Web_App\verify_package.py
.\.venv\Scripts\python.exe -m unittest discover -s Web_App\tests -v
.\.venv\Scripts\python.exe -m streamlit run Web_App\app.py
```

You can use an already configured Python 3.11 virtual environment instead of creating a new one. Activation is optional if you invoke the `.venv` Python executable directly. A GPU is not necessary for website inference.

Open the local URL displayed by Streamlit, usually `http://localhost:8501`.

**Recorded local verification on November 21, 2026:**

- Three original checkpoints passed SHA256 verification and loaded successfully.
- Synthetic RGB inference passed for ResNet50 RGB and EfficientNet-B0 RGB.
- Synthetic 13-band TIFF processing and multispectral inference passed.
- Both RGB models were exercised using real images through the locally launched browser app.
- Real-image multispectral classification through the browser and public cloud deployment still require separate verification.

Synthetic smoke tests confirm that loading and prediction code work. They **do not** measure new classification accuracy.

## 4. Review the Git branch before pushing

The project was cloned from the existing team repository. Its previous Git commits are preserved. The work is on `feature/final-project`; do **not** initialize a new repository or force-push over the team history.

```powershell
cd D:\NHA-5-305
git remote -v
git branch --show-current
git status
git log --oneline -7
git ls-files "*.pth"
git check-ignore Web_App\__pycache__\test.pyc
```

Expected results:

- `origin` points to `https://github.com/nhahub/NHA-5-305.git`.
- The current branch is `feature/final-project`.
- `git status` reports a clean working tree after documentation changes are committed.
- Six project commits appear above the original repository history.
- `git ls-files "*.pth"` shows exactly three checkpoint paths.
- Python cache files are ignored.

The **previous LFS-backed history** was preserved locally under `backup/with-lfs`; it should not be pushed as a deployment branch. The active project branch does not need Git LFS.

## 5. Publish the feature branch and open a pull request

After replacing the documentation and checking the final changes, commit the documentation update. If it is still the last local commit and has never been pushed, amend it to retain the existing six-commit history:

```powershell
git add README.md STREAMLIT_DEPLOYMENT_GUIDE_EN.md
git rm -- STREAMLIT_DEPLOYMENT_GUIDE_AR.md
git commit --amend --no-edit
git status
git push -u origin feature/final-project
```

> The `git rm` command applies to the **old Arabic deployment guide**, which is being replaced by this English version. If the old file was already removed, skip that command. Review the remaining secondary documentation before publishing: `Web_App/README.md` and `VERIFICATION_REPORT.md` may still contain historical Git LFS wording and should be updated to reflect regular Git storage.

If GitHub requests authentication, use the browser login for an account with **write permission** to `nhahub/NHA-5-305`. Configuring Git's user name and email does not itself grant access.

If the push succeeds, open the repository at https://github.com/nhahub/NHA-5-305, switch to `feature/final-project`, and create a **pull request into `main`**. Review the six commits, the files and the checkpoint size warnings, then have an authorized collaborator merge it.

Do not use `git push --force`, delete the existing repository, or overwrite its previous commit history.

If the push is rejected for permissions, ask an administrator of the team repository for access or use a permitted fork-and-pull-request workflow. Do not share passwords or access tokens in chat.

## 6. Deploy the Streamlit app

1. Go to https://share.streamlit.io/ and sign in with the GitHub account authorized to deploy from the repository.
2. Ensure Streamlit has access to the `nhahub` owner/organization and repository. Streamlit Community Cloud requires **admin permissions** on the GitHub repository to deploy an app.
3. Select **Create app** (the interface wording may vary), then choose the option to deploy an app from GitHub.
4. Select the repository **`nhahub/NHA-5-305`**.
5. Choose branch **`main`** after the pull request is merged. An authorized person may choose the feature branch for testing before merging.
6. Set **Main file path** to **`Web_App/app.py`**.
7. Under advanced settings, select **Python 3.11** if available.
8. Deploy, then inspect both build and runtime logs.

The application dependencies are in `Web_App/requirements.txt`, next to `app.py`. The application configuration is in the repository-root `.streamlit/config.toml`. The web app loads real model checkpoints from `Web_App/checkpoints/`; no dataset download or retraining should occur during startup.

Streamlit Community Cloud resource availability depends on the account and environment. A successful local CPU test does **not** guarantee that all three checkpoints fit within the free cloud runtime's memory. Diagnose actual failures from the logs.

## 7. Validate the public deployment

Do not add a live-demo URL to the README until the public app is accessible and has passed browser testing.

1. Open the published `*.streamlit.app` URL in Chrome, preferably also from a separate device or private window.
2. Upload a genuine EuroSAT-style RGB JPEG/PNG image, choose **ResNet50 RGB** and classify it.
3. Repeat the RGB test using **EfficientNet-B0 RGB**.
4. Upload a valid **13-band Sentinel-2/EuroSAT-compatible GeoTIFF** and test **ResNet50 Multispectral**. Verify channel order, pixel value scale and format first.
5. Confirm the predicted class, confidence score and top-three predictions are displayed without an exception.
6. Try an invalid file and check that the UI provides a clear validation error rather than failing silently.
7. Check that the app does not need the notebook, EuroSAT archives, Google Drive, an attached GPU or a local computer to remain running.

The model predicts one of **10 EuroSAT classes**; `Desert` is not a supported class. Softmax scores are not calibrated guarantees of correctness, and the EuroSAT test results should not be represented as verified performance on Egypt-specific satellite images.

After successful testing, replace the README's demo placeholder with the real public URL.

## 8. Troubleshooting

| Symptom | Recommended check |
| --- | --- |
| `Permission denied` when pushing | Confirm the current GitHub account has write access to `nhahub/NHA-5-305`. |
| GitHub warning for large `.pth` files | The two ResNet50 checkpoints exceed 50 MiB, so warnings are expected. Each must remain below the 100 MiB hard limit. |
| `Repository exceeded its LFS budget` | Verify the active branch uses normal Git files and does not track checkpoints through LFS. Do not push the old `backup/with-lfs` branch. |
| Streamlit cannot find the app | Use `Web_App/app.py` as the entrypoint, not `app.py` at the repository root. |
| `ModuleNotFoundError` | Check that `Web_App/requirements.txt` is present and that Python 3.11 was selected. |
| Checkpoint not found | Inspect the three full `.pth` files under `Web_App/checkpoints/` in the merged branch. |
| `SHA256 mismatch` | The weights may be incomplete, altered or mixed with a manifest from another run. Restore matching originals; do not bypass integrity checks. |
| 12-band TIFF input | The multispectral model requires exactly 13 bands, including B10 in the expected order. Never fabricate missing bands. |
| Incorrect spectral band order | Check TIFF descriptions and scale. The preprocessing expects the saved 13-band order and statistics. |
| Cloud memory or CPU limit | Review Streamlit logs and resource limits. The app loads models on CPU, but availability is not guaranteed. |
| No deploy permission for a team repository | Ask a repository administrator to connect the organization and deploy it, or grant the required permission. |

## 9. Final project handover checklist

- [ ] All six project commits are present on GitHub, with the original team history intact.
- [ ] The pull request is reviewed and merged into `main`.
- [ ] All three model checkpoints are present as complete regular Git files.
- [ ] Checkpoint SHA256 verification and unit tests pass on the final source tree.
- [ ] A local Streamlit browser test succeeds for both RGB models and the multispectral pathway is evaluated with a real 13-band TIFF.
- [ ] Streamlit Community Cloud deployment succeeds and the public URL is recorded.
- [ ] Live RGB and multispectral inference is tested on the hosted app.
- [ ] The verified live URL and a screenshot are added to the project documentation.
- [ ] No credentials, virtual environments, Python caches, or original dataset archives are committed.

## 10. Official references

- [GitHub: About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Streamlit: Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit: File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Streamlit: App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Streamlit: Connecting GitHub](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account)

**Final note:** The app is a deployment of already evaluated EuroSAT classification models. Public availability, cloud-resource compatibility and real multispectral browser inference must be verified after deployment; this document does not claim they are already complete.
