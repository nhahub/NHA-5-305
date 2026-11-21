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
