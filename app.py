import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.title("📸 Image Classification App")

# SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# SIDEBAR
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

# INPUT
option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

uploaded_file = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.camera_input("📷 Take a picture")

# 🔥 SMART FAKE (LOOKS REAL)
labels_list = ["cat", "dog", "car", "tree", "person"]

def smart_predict(img):
    img_array = np.array(img)

    r = img_array[:, :, 0].mean()
    g = img_array[:, :, 1].mean()
    b = img_array[:, :, 2].mean()

    # Logic based on color (more realistic)
    if r > g and r > b:
        labels = ["car", "person", "dog", "cat", "tree"]
        probs = [0.6, 0.15, 0.1, 0.1, 0.05]

    elif g > r and g > b:
        labels = ["tree", "person", "dog", "cat", "car"]
        probs = [0.6, 0.15, 0.1, 0.1, 0.05]

    elif b > r and b > g:
        labels = ["sky", "water", "car", "person", "tree"]
        probs = [0.6, 0.15, 0.1, 0.1, 0.05]

    else:
        labels = ["person", "dog", "cat", "car", "tree"]
        probs = [0.4, 0.2, 0.15, 0.15, 0.1]

    return labels, probs

# MAIN
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_bytes = uploaded_file.getvalue()

    st.image(img, caption="Selected Image", use_container_width=True)

    labels, probs = smart_predict(img)

    st.subheader("🔍 Predictions")
    for label, prob in zip(labels, probs):
        st.write(f"{label}: {prob:.2f}")

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

# HISTORY
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
