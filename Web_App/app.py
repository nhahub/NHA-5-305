"""Streamlit Community Cloud frontend for the three verified EuroSAT checkpoints.

Deploy from the repository root with:
    streamlit run Web_App/app.py

All training and preprocessing metadata come from the actual saved checkpoints.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from inference import (
    BAND_ORDER,
    MODEL_MS,
    MODEL_RGB_EFFICIENT,
    MODEL_RGB_RESNET,
    get_manifest,
    predict_ms,
    predict_rgb,
)

RGB_MAX_SIZE = 10 * 1024 * 1024
MS_MAX_SIZE = 25 * 1024 * 1024

CLASS_DESCRIPTIONS = {
    "AnnualCrop": "Annual cropland / محاصيل سنوية",
    "Forest": "Forest / غابات",
    "HerbaceousVegetation": "Herbaceous vegetation / نباتات عشبية",
    "Highway": "Highway / طرق رئيسية",
    "Industrial": "Industrial area / مناطق صناعية",
    "Pasture": "Pasture / مراعي",
    "PermanentCrop": "Permanent cropland / محاصيل دائمة",
    "Residential": "Residential area / مناطق سكنية",
    "River": "River / أنهار",
    "SeaLake": "Sea or lake / بحيرات وبحار",
}


@st.cache_data(show_spinner=False)
def metrics_by_model() -> dict[str, dict]:
    """Use final-run manifest for displayed scores; no retraining or guessed metrics."""
    return {row["Model"]: row for row in get_manifest().get("metrics", [])}


def load_rgb_upload(upload) -> Image.Image:
    if upload is None:
        raise ValueError("Please upload a JPG or PNG satellite image first.")
    if len(upload.getvalue()) > RGB_MAX_SIZE:
        raise ValueError("RGB file is too large (maximum 10 MB).")
    try:
        with Image.open(io.BytesIO(upload.getvalue())) as source:
            if source.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPG or PNG images are accepted for RGB models.")
            if source.width * source.height > 16_000_000:
                raise ValueError("RGB image is too large; use a small satellite patch.")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("This is not a readable JPG or PNG image.") from exc


def classify_ms_upload(upload, verified: bool) -> dict:
    if upload is None:
        raise ValueError("Please upload a 13-band GeoTIFF file first.")
    if not verified:
        raise ValueError("Confirm the sensor, value scale and band order before running the 13-band model.")
    content = upload.getvalue()
    if len(content) > MS_MAX_SIZE:
        raise ValueError("TIFF file is too large for this public demo (maximum 25 MB).")
    if not content:
        raise ValueError("The uploaded TIFF is empty.")
    # rasterio accepts a filename. Use a short-lived local file, then remove it.
    suffix = Path(upload.name).suffix.lower()
    if suffix not in (".tif", ".tiff"):
        raise ValueError("Only .tif and .tiff files are accepted for multispectral inference.")
    path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="eurosat_ms_", suffix=suffix, delete=False) as stream:
            path = Path(stream.name)
            stream.write(content)
        return predict_ms(path, confirm_unlabeled_band_order=True)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def display_prediction(result: dict) -> None:
    predicted = str(result["prediction"])
    score = float(result["model_score"])
    st.success(f"Predicted land type: {predicted}", icon="✅")
    st.metric("Top model score (not calibrated)", f"{score:.1%}")
    st.caption(CLASS_DESCRIPTIONS.get(predicted, predicted))
    st.subheader("Top 3 predictions")
    for rank, row in enumerate(result["top3"], 1):
        name = str(row["class_name"])
        value = float(row["score"])
        st.write(f"**{rank}. {name}** — {value:.1%}")
        st.progress(min(1.0, max(0.0, value)))
    st.caption("Softmax scores are not calibrated probabilities. Predictions are most reliable on EuroSAT-style satellite patches.")


def model_details(name: str) -> None:
    info = get_manifest()["models"][name]
    metrics = metrics_by_model().get(name, {})
    with st.expander("Model information and benchmark", expanded=False):
        st.write(f"**Architecture:** {info['architecture']}")
        st.write(f"**Input:** {info['input_channels']} channels, {info['image_size']} × {info['image_size']} after preprocessing")
        if metrics:
            st.write(f"**Held-out EuroSAT accuracy:** {100 * float(metrics['Test accuracy']):.2f}% on {int(metrics['Test images']):,} images")
            st.write(f"**Macro F1:** {100 * float(metrics['Macro F1']):.2f}%")
        st.caption("Benchmarks from the final trained run, not a measurement of this uploaded image.")


def main() -> None:
    st.set_page_config(
        page_title="Land Type Classification | Satellite AI",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1150px;}
      .hero-title {font-size:2.2rem; font-weight:800; letter-spacing:-.035em;}
      .hero-sub {color:#9eb4bd; margin:0 0 1.1rem 0;}
      .pill {display:inline-block; border:1px solid #28514e; background:#13382f;
             border-radius:999px; padding:4px 12px; color:#93e7ce; font-size:.8rem;}
      .muted {color:#9ab0b7; font-size:.91rem;}
    </style>
    """, unsafe_allow_html=True)

    manifest = get_manifest()
    with st.sidebar:
        st.markdown("### 🛰️ Satellite AI")
        st.caption("EuroSAT land-cover classification")
        st.divider()
        st.markdown("**Three trained models**")
        for model_name in (MODEL_RGB_RESNET, MODEL_RGB_EFFICIENT, MODEL_MS):
            row = metrics_by_model().get(model_name)
            if row:
                st.write(f"{model_name}: **{100 * float(row['Test accuracy']):.2f}%**")
        st.divider()
        st.markdown("**Training dataset**")
        st.write("EuroSAT · 27,000 images · 10 classes")
        st.write("Train: 18,900 · Validation: 4,050 · Test: 4,050")
        st.caption(f"Run: {manifest.get('run_id', 'unknown')}")
        st.divider()
        st.caption("Inference-only web app. No retraining or satellite data download occurs here.")

    st.markdown('<div class="pill">DEEP LEARNING · SATELLITE IMAGERY</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Land Type Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload a satellite image, choose a trained model, and predict the land-cover category. ' 
        'ارفع صورة قمر صناعي لمعرفة نوع الأرض.</div>', unsafe_allow_html=True
    )

    tab_rgb, tab_ms, tab_guide = st.tabs(["📷 RGB Image (JPG / PNG)", "🛰️ Multispectral (13-band TIFF)", "ℹ️ About / Input Guide"])
    with tab_rgb:
        st.subheader("Classify an RGB satellite image")
        st.caption("Use small satellite patches similar to EuroSAT. Ordinary photographs are out of scope.")
        left, right = st.columns([1, 1], gap="large")
        with left:
            with st.form("rgb_classification", clear_on_submit=False):
                upload = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"], key="rgb_upload")
                model_name = st.selectbox("Choose RGB model", [MODEL_RGB_RESNET, MODEL_RGB_EFFICIENT])
                submitted = st.form_submit_button("🔎 Classify Image", type="primary", use_container_width=True)
            if upload is not None:
                try:
                    st.image(load_rgb_upload(upload), caption="Uploaded satellite patch", use_container_width=True)
                except ValueError as exc:
                    st.warning(str(exc))
            model_details(model_name)
        with right:
            st.markdown("#### Classification result")
            if submitted:
                try:
                    image = load_rgb_upload(upload)
                    with st.spinner("Running the trained model on CPU..."):
                        result = predict_rgb(image, model_name)
                    display_prediction(result)
                except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
                    st.error(f"Could not classify image: {exc}")
            else:
                st.info("Upload an RGB satellite image and press **Classify Image** to see the prediction.")

    with tab_ms:
        st.subheader("Classify a 13-band Sentinel-2 GeoTIFF")
        st.warning("This is **not** a regular RGB photo. Only 13-band TIFFs with verified EuroSAT-compatible band order and pixel-value scale are suitable.", icon="⚠️")
        lcol, rcol = st.columns([1, 1], gap="large")
        with lcol:
            with st.form("ms_classification", clear_on_submit=False):
                ms_upload = st.file_uploader("Upload a 13-band .tif / .tiff", type=["tif", "tiff"], key="ms_upload")
                verified = st.checkbox(
                    "I verified the Sentinel-2 13-band order AND pixel-value scale. "
                    "For TIFFs without band descriptions, I also verified channel order."
                )
                ms_submitted = st.form_submit_button("🛰️ Classify GeoTIFF", type="primary", use_container_width=True)
            st.caption("Required order: " + ", ".join(BAND_ORDER))
            model_details(MODEL_MS)
        with rcol:
            st.markdown("#### Multispectral result")
            if ms_submitted:
                try:
                    with st.spinner("Reading TIFF and running the 13-band model on CPU..."):
                        result = classify_ms_upload(ms_upload, verified)
                    display_prediction(result)
                except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
                    st.error(f"Could not classify GeoTIFF: {exc}")
            else:
                st.info("Upload a compatible 13-band GeoTIFF, confirm its format, then press **Classify GeoTIFF**.")

    with tab_guide:
        st.subheader("What does this model recognize?")
        grid = st.columns(2)
        for i, (name, caption) in enumerate(CLASS_DESCRIPTIONS.items()):
            grid[i % 2].write(f"**{name}** — {caption.split(' / ')[-1]}")
        st.divider()
        st.markdown("**Important limitations**")
        st.write("- `Desert` is **not** one of the 10 classes. Unknown land types can be forced into a known category.")
        st.write("- These results use an image-level stratified EuroSAT test split, **not** an independent geographic holdout.")
        st.write("- RGB inputs use ImageNet normalization and 224×224 resizing; the 13-band model uses training-only band statistics saved in its checkpoint.")
        st.write("- An external 13-band TIFF might have a different band order, missing B10, or a different radiometric scale; do not assume compatibility.")
        st.write("- The percentages shown with predictions are uncalibrated softmax scores, not guarantees of correctness.")
        st.caption("Source: final_run_audit.json and deployment_manifest.json from the executed graduation-project notebook.")

    st.divider()
    st.caption("Graduation Project · Land Type Classification Using Deep Learning and Satellite Imagery · Powered by PyTorch and Streamlit")


if __name__ == "__main__":
    main()
