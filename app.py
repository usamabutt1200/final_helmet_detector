import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Vanguard Helmet Detection Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: Inter, system-ui, sans-serif;
    }
    div[data-testid="stSidebar"] {
        background-color: #121620;
        border-right: 1px solid #1E293B;
    }
    .metric-card, .control-card {
        background: linear-gradient(145deg, #161D2A 0%, #0F131D 100%);
        border: 1px solid #243046;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.35);
    }
    .metric-card:hover {
        border-color: #00F5D4;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.12);
    }
    .glow-text {
        color: #00F5D4;
        text-shadow: 0 0 15px rgba(0, 245, 212, 0.25);
        font-weight: 800;
        font-size: 2rem;
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    """Load the custom helmet detector once and cache it."""
    model_path = "final_helmet_detector_v8.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' was not found. "
            "Place it in the same folder as app.py."
        )

    return YOLO(model_path)


try:
    neural_engine = load_model()
    model_ready = True
except Exception as error:
    neural_engine = None
    model_ready = False
    model_error = str(error)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown(
    "<h1 class='glow-text' style='text-align:center;'>VANGUARD OS</h1>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
