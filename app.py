import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests

# 🔑 USE STREAMLIT SECRETS (SAFE)
API_TOKEN = st.secrets["HF_TOKEN"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

st.title("📸 Image Classification App")

# ✅ SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

if st.sidebar.button("🆕 New Chat"):
    st.session_state.history = []
    st.session_state.last_image = None
    st.rerun()

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    st.rerun()

if st.sidebar.button("🗑️ Delete History"):
    st.session_state.history = []
    st.sidebar.success("History deleted!")
    st.rerun()

show_history = st.sidebar.checkbox("📜 Show History")

# =========================
# INPUT METHOD
# =========================
option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

uploaded_file = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

elif option == "Use Camera":
    uploaded_file = st.camera_input("📷 Take a picture")

# =========================
# REAL PREDICTION FUNCTION
# =========================
def real_predict(img_bytes):
    for i in range(3):  # retry 3 times
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=img_bytes,
                timeout=30
            )

            if response.status_code == 503:
                return ["Model loading... Try again"], [0]

            result = response.json()

            if isinstance(result, list):
                labels = [item["label"] for item in result[:5]]
                probs = [item["score"] for item in result[:5]]
                return labels, probs

        except requests.exceptions.RequestException:
            pass

    return ["Network Error / Try Again"], [0]
# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_bytes = uploaded_file.getvalue()

    st.image(img, caption="Selected Image", use_container_width=True)

    # 🔍 REAL AI PREDICTION
    with st.spinner("Analyzing image..."):
        labels, probs = real_predict(img_bytes)

    st.subheader("🔍 Predictions")
    for label, prob in zip(labels, probs):
        st.write(f"{label}: {prob:.2f}")

    # 📊 Bar chart
    st.subheader("📊 Analysis")
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    ax.bar(labels, probs, color=colors)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # Save history
    if st.session_state.last_image != img_bytes:
        st.session_state.history.append({
            "image": img,
            "labels": labels,
            "probs": probs
        })
        st.session_state.last_image = img_bytes

# =========================
# HISTORY
# =========================
if show_history:
    st.subheader("📜 History")

    for item in st.session_state.history[::-1]:
        st.image(item["image"], width=200)

        for label, prob in zip(item["labels"], item["probs"]):
            st.write(f"{label}: {prob:.2f}")

        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        ax.bar(item["labels"], item["probs"], color=colors)
        plt.xticks(rotation=30)
        st.pyplot(fig)
