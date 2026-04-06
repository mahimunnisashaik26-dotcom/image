import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load model
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

st.title("📸 Image Classification App")

# ✅ SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# =========================
# ✅ SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

if st.sidebar.button("🆕 New Chat"):
    st.session_state.history = []
    st.session_state.last_image = None
    st.rerun()

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []

if st.sidebar.button("🗑️ Delete History"):
    st.session_state.history = []
    st.sidebar.success("History deleted!")

show_history = st.sidebar.checkbox("📜 Show History")

# =========================
# ✅ INPUT METHOD
# =========================
option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

uploaded_file = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

elif option == "Use Camera":
    uploaded_file = st.camera_input("📷 Take a picture")

# =========================
# ✅ MAIN LOGIC
# =========================
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    # Convert image to bytes (for comparison)
    img_bytes = uploaded_file.getvalue()

    st.image(img, caption="Selected Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]

    labels = [i[1] for i in decoded]
    probs = [i[2] for i in decoded]

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

    # ✅ ADD ONLY IF NEW IMAGE
    if st.session_state.last_image != img_bytes:
        st.session_state.history.append({
            "image": img,
            "labels": labels,
            "probs": probs
        })
        st.session_state.last_image = img_bytes

# =========================
# ✅ HISTORY
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