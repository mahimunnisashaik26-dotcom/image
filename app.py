import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

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
# FAKE PREDICTION (NO TF)
# =========================
labels_list = ["cat", "dog", "car", "tree", "person"]

def fake_predict():
    probs = np.random.rand(5)
    probs = probs / probs.sum()
    return labels_list, probs

# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_bytes = uploaded_file.getvalue()

    st.image(img, caption="Selected Image", use_container_width=True)

    labels, probs = fake_predict()

    st.subheader("🔍 Predictions")
    for label, prob in zip(labels, probs):
        st.write(f"{label}: {prob:.2f}")

    # Bar chart
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
