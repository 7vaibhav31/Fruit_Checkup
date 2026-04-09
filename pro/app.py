"""
streamlit.py  (FreshVision AI)
------------------------------
Professional Streamlit frontend for the Fruit Freshness Classifier.
Uses CNN_MDOEL(FINE_TUNE).keras  +  Grad-CAM explainability.

Run:
    cd d:/CNN_PROJECT/pro
    streamlit run streamlit.py
"""

import os
import pathlib
import time

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from gradcam import build_grad_model, get_gradcam_from_array

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "FreshVision AI",
    page_icon  = "🍎",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
CSS_PATH = pathlib.Path(__file__).parent / "style.css"
with open(CSS_PATH, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "CNN_MDOEL(FINE_TUNE).keras"

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

# Pretty labels shown in the UI
PRETTY = {
    "freshapples"  : "🍎 Fresh Apple",
    "freshbanana"  : "🍌 Fresh Banana",
    "freshoranges" : "🍊 Fresh Orange",
    "rottenapples" : "🍎 Rotten Apple",
    "rottenbanana" : "🍌 Rotten Banana",
    "rottenoranges": "🍊 Rotten Orange",
}

FRESH_CLASSES  = {"freshapples", "freshbanana", "freshoranges"}

# ── Model loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model      = load_model(str(MODEL_PATH))
    grad_model = build_grad_model(model, last_conv_layer_name="out_relu")
    return model, grad_model

# ── Helper: preprocess PIL image → normalised numpy array ────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape (1, 224, 224, 3)

# ── Helper: HTML for a custom progress / probability bar ─────────────────────
def prob_bar(name: str, prob: float) -> str:
    pct  = prob * 100
    fill = f'<div class="prob-bar-fill" style="width:{pct:.1f}%"></div>'
    return (
        f'<div class="prob-row">'
        f'  <span class="prob-name">{name}</span>'
        f'  <div class="prob-bar-bg">{fill}</div>'
        f'  <span class="prob-pct">{pct:.1f}%</span>'
        f'</div>'
    )

# ── Helper: render full result UI ─────────────────────────────────────────────
def show_results(pil_img, model, grad_model):
    arr         = preprocess(pil_img)
    preds       = model.predict(arr, verbose=0)[0]
    pred_idx    = int(np.argmax(preds))
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = float(preds[pred_idx]) * 100
    is_fresh    = pred_class in FRESH_CLASSES

    status_color = "fresh" if is_fresh else "rotten"
    status_emoji = "✅" if is_fresh else "⚠️"
    status_label = "FRESH" if is_fresh else "ROTTEN"

    # ── Result card ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-card result-{status_color}">
          <div class="result-header">
            <span class="result-emoji">{status_emoji}</span>
            <p class="result-title result-{status_color}-text">
              {PRETTY[pred_class]} — {status_label}
            </p>
          </div>

          <div class="confidence-wrap">
            <div class="confidence-label">
              Confidence &nbsp;·&nbsp; <strong style="color:var(--text)">{confidence:.1f}%</strong>
            </div>
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill-{status_color}"
                   style="width:{confidence:.1f}%"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Per-class probability breakdown ─────────────────────────────────
    with st.expander("📊 Full class probability breakdown", expanded=False):
        sorted_idx = np.argsort(preds)[::-1]
        html_bars  = "".join(
            prob_bar(PRETTY.get(CLASS_NAMES[i], CLASS_NAMES[i]), float(preds[i]))
            for i in sorted_idx
        )
        st.markdown(html_bars, unsafe_allow_html=True)

    # ── Grad-CAM section ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="gradcam-panel">
          <p class="gradcam-title">🔬 Grad-CAM Explainability</p>
          <p class="gradcam-desc">
            Grad-CAM highlights the image regions the model focused on
            when making its decision. <span style="color:#d32f2f; font-weight:bold;">Red/warm areas</span> show where it focuses,
            while <span style="color:#1976d2; font-weight:bold;">blue regions</span> indicate areas it ignores.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Generating Grad-CAM heatmap …"):
        time.sleep(0.2)          # tiny pause so spinner is visible
        orig, heatmap, overlay, pred_c, conf = get_gradcam_from_array(
            arr, model, grad_model, CLASS_NAMES
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(orig,    caption="Original Image",      use_container_width=True, clamp=True)
    with col_b:
        import matplotlib.cm as cm
        hm_colored = cm.jet(heatmap)[:, :, :3]          # H×W×3 float
        st.image(hm_colored, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
    with col_c:
        st.image(overlay, caption=f"Overlay — {PRETTY.get(pred_c,'')}", use_container_width=True, clamp=True)

    # ── Advisory message ─────────────────────────────────────────────────
    if is_fresh:
        advice = "✅ This fruit appears <strong>fresh</strong> and is safe to consume."
    else:
        advice = ("⚠️ This fruit appears <strong>rotten</strong>. "
                  "Consuming it may cause health issues. Please discard it.")

    st.markdown(
        f'<div class="info-box">{advice}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        '  <div class="sidebar-logo-text">🍎 FreshVision AI</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section-title">📦 Supported Classes</div>',
        unsafe_allow_html=True,
    )
    chips = {
        "freshapples"  : "chip-fresh",
        "freshbanana"  : "chip-fresh",
        "freshoranges" : "chip-fresh",
        "rottenapples" : "chip-rotten",
        "rottenbanana" : "chip-rotten",
        "rottenoranges": "chip-rotten",
    }
    chips_html = "".join(
        f'<span class="class-chip {v}">{PRETTY.get(k, k)}</span>'
        for k, v in chips.items()
    )
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section-title">🧠 Model Info</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.82rem;color:var(--text-muted);line-height:1.75">
          <b style="color:var(--text)">Architecture</b> &nbsp;MobileNetV2 + Custom Head<br>
          <b style="color:var(--text)">Input size</b> &nbsp;&nbsp;&nbsp;&nbsp;224 × 224 px<br>
          <b style="color:var(--text)">Classes</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6<br>
          <b style="color:var(--text)">Val. Accuracy</b> &nbsp;96.8 %<br>
          <b style="color:var(--text)">Explainability</b> Grad-CAM<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section-title">ℹ️ How to Use</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.82rem;color:var(--text-muted);line-height:1.8">
          1. Choose <b>Upload</b> or <b>Camera</b> tab<br>
          2. Provide an image of a fruit<br>
          3. View the prediction &amp; confidence<br>
          4. Explore the Grad-CAM heatmap<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#   HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">Powered by MobileNetV2 · Grad-CAM</div>
      <h1 class="hero-title">FreshVision AI</h1>
      <p class="hero-subtitle">
        Instantly detect whether your fruit is <strong>fresh</strong> or
        <strong>rotten</strong> — with explainable AI heatmaps showing
        exactly what the model is looking at.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Stat cards ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-number">6</div>
        <div class="stat-label">Fruit Classes</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">96.8%</div>
        <div class="stat-label">Val. Accuracy</div>
      </div>
      <div class="stat-card">
        <div class="stat-number">Grad-CAM</div>
        <div class="stat-label">Explainability</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#   LOAD MODEL (with spinner)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading AI model …  This may take a few seconds."):
    model, grad_model = load_resources()

# ══════════════════════════════════════════════════════════════════════════════
#   INPUT: UPLOAD  /  CAMERA  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_upload, tab_camera = st.tabs(["📁  Upload Image", "📷  Use Camera"])

# ── Tab 1: Upload ──────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        """
        <div class="upload-section">
          <div class="upload-icon">📤</div>
          <p class="upload-title">Drag &amp; drop or browse</p>
          <p class="upload-hint">Supports JPG · JPEG · PNG · WEBP · BMP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        label       = "Choose an image",
        type        = ["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility = "collapsed",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)

        col_img, col_res = st.columns([1, 2], gap="large")
        with col_img:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

        with col_res:
            with st.spinner("Analyzing fruit …"):
                show_results(pil_img, model, grad_model)

# ── Tab 2: Camera ──────────────────────────────────────────────────────────
with tab_camera:
    st.markdown(
        """
        <div class="upload-section">
          <div class="upload-icon">📷</div>
          <p class="upload-title">Click a photo of your fruit</p>
          <p class="upload-hint">
            Make sure the fruit is well-lit and centred in the frame.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_img = st.camera_input(
        label = "Take a photo",
        label_visibility = "collapsed",
    )

    if camera_img is not None:
        pil_img = Image.open(camera_img)

        col_img, col_res = st.columns([1, 2], gap="large")
        with col_img:
            st.image(pil_img, caption="Camera capture", use_container_width=True)

        with col_res:
            with st.spinner("Analyzing fruit …"):
                show_results(pil_img, model, grad_model)

# ══════════════════════════════════════════════════════════════════════════════
#   FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="app-footer">
      FreshVision AI &nbsp;·&nbsp; Built with TensorFlow &amp; Streamlit
      &nbsp;·&nbsp; Model: MobileNetV2 Fine-Tuned
      &nbsp;·&nbsp; Explainability: Grad-CAM
    </div>
    """,
    unsafe_allow_html=True,
)
