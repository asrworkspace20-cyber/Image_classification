import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

st.set_page_config(page_title="Image Classifier", page_icon="🖼️")


@st.cache_resource
def load_model():
    # Downloads pretrained ImageNet weights once, then caches across sessions.
    # No training needed — this model already knows 1,000 real-world categories.
    return MobileNetV2(weights="imagenet")


model = load_model()

st.title("🖼️ Image Classifier")
st.write(
    "Upload any photo — this model recognizes **1,000 real-world categories**: "
    "animals, vehicles, food, household objects, plants, and much more."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Your upload", use_column_width=True)

    # MobileNetV2 expects 224x224 images
    resized = img.resize((224, 224))
    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # MobileNetV2's own normalization

    with st.spinner("Classifying..."):
        preds = model.predict(img_array, verbose=0)

    # Decode into human-readable labels, top 5 guesses
    decoded = decode_predictions(preds, top=5)[0]

    st.subheader("Predictions")
    for _, label, confidence in decoded:
        readable_label = label.replace("_", " ").title()
        st.write(f"**{readable_label}**")
        st.progress(float(confidence))
        st.caption(f"{confidence * 100:.1f}% confidence")
