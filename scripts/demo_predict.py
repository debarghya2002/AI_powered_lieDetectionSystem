"""Streamlit file-picker demo for an already-trained MU3D model.

Run from the project root:
    python -m streamlit run scripts/demo_predict.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.file_predictor import VideoFilePredictor


@st.cache_resource(show_spinner=False)
def load_predictor(bundle_path: str, device: str):
    return VideoFilePredictor(bundle_path, device=device)


@st.cache_data(show_spinner=False)
def load_metadata(path: str):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: str):
    with open(path) as file:
        return json.load(file)


def save_upload(upload) -> Path:
    suffix = Path(upload.name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(upload.name).stem)
    digest = hashlib.sha256(upload.getvalue()).hexdigest()[:12]
    destination = ROOT / "outputs" / "demo_uploads" / f"{safe_stem}_{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(upload.getvalue())
    return destination


def matching_label(metadata: pd.DataFrame, filename: str):
    video_id = Path(filename).stem
    row = metadata[metadata.video_id.astype(str) == video_id]
    return None if row.empty else row.iloc[0]


st.set_page_config(page_title="MU3D Video Prediction Demo", page_icon="🎬", layout="wide")
st.title("MU3D Deception Detection — Video Prediction Demo")
st.info("Prediction-only mode: the trained model is loaded; this screen never trains or changes it.")

default_bundle = ROOT / "outputs" / "inference_bundle.pt"
with st.sidebar:
    st.header("Model")
    bundle_text = st.text_input("Prediction bundle", value=str(default_bundle))
    device = st.selectbox("Device", ("auto", "cpu", "cuda"), index=0)
    st.caption("Use CPU if CUDA is not configured in your environment.")

bundle_path = Path(bundle_text)
if not bundle_path.exists():
    st.warning("Create the prediction bundle once. This packages the existing checkpoint; it does not train it.")
    st.code("python scripts/create_inference_bundle.py", language="powershell")
    st.stop()

uploaded = st.file_uploader("Choose a video sample", type=["mp4", "wmv", "avi", "mov", "mkv"])
if uploaded is None:
    st.caption("Choose an MU3D sample such as `WM027_1PT.wmv` to also display its known label.")
    st.stop()

video_path = save_upload(uploaded)
st.video(uploaded.getvalue())
metadata = load_metadata(str(ROOT / "data" / "master_metadata.csv"))
known = matching_label(metadata, uploaded.name)
if known is not None:
    known_name = "Truth" if int(known.label) == 1 else "Lie"
    st.caption(f"Known MU3D label found: **{known_name}** (`{known.video_id}`).")
else:
    st.caption("No known MU3D label for this filename. A prediction will be shown, but accuracy cannot be calculated.")

if st.button("Run prediction", type="primary", use_container_width=True):
    try:
        with st.spinner("Extracting face and audio features, then running the saved model..."):
            result = load_predictor(str(bundle_path), device).predict(video_path)
    except Exception as exc:
        st.error(f"Prediction could not run: {exc}")
        st.stop()

    if result.status != "ok":
        st.warning(f"No score shown: {result.status}")
        st.stop()

    predicted_label = int(result.truth_probability >= 0.5)
    predicted_name = "Truth-like" if predicted_label else "Deception-like"
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", predicted_name)
    col2.metric("Deception likelihood", f"{result.deception_probability:.1%}")
    col3.metric("Truth probability", f"{result.truth_probability:.1%}")
    st.caption("These are model probabilities from controlled MU3D training data, not factual lie determinations.")

    if known is not None:
        correct = predicted_label == int(known.label)
        target = "Truth" if int(known.label) else "Lie"
        if correct:
            st.success(f"Known label: {target}. This single clip was classified correctly.")
        else:
            st.error(f"Known label: {target}. This single clip was classified incorrectly.")
        st.caption("For one labelled clip, the meaningful result is correct/incorrect—not a standalone accuracy percentage.")

    metrics_path = ROOT / "outputs" / "loso_metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(str(metrics_path))
        st.subheader("Stored full-model LOSO results")
        labels = [("Accuracy", "accuracy"), ("Precision", "precision"),
                  ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]
        columns = st.columns(len(labels))
        for column, (label, key) in zip(columns, labels):
            column.metric(label, f"{metrics[key]:.1%}")
        st.caption("These are the saved aggregate results across all 320 MU3D clips, not metrics recalculated from this one video.")
