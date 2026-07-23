import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

# ---------- Simple custom styling ----------
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


model = load_model()

# ---------- Header ----------
st.title("🖼️ What's In This Photo?")
st.write(
    "Upload a photo or take one with your camera, and I'll tell you what's in it. "
    "Recognizes **1,000 categories** — animals, food, vehicles, plants, household items, and more."
)

with st.expander("ℹ️ Tips for best results"):
    st.markdown(
        "- Works best with **one clear subject** per photo\n"
        "- Good lighting helps a lot\n"
        "- Very cluttered or busy scenes may confuse it\n"
        "- It was trained on everyday objects — very obscure or specialized items may not be recognized"
    )

st.divider()

# ---------- Input method ----------
# A changing "uploader_key" forces Streamlit to treat the uploader/camera as brand
# new widgets when reset, which is what actually clears a previously uploaded photo.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

tab1, tab2 = st.tabs(["📁 Upload a photo", "📷 Use camera"])

image_source = None
with tab1:
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input(
        "Take a photo",
        label_visibility="collapsed",
        key=f"camera_{st.session_state.uploader_key}",
    )
    if camera_file is not None:
        image_source = Image.open(camera_file)

# ---------- Prediction ----------
if image_source is not None:
    image_source = image_source.convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_source, caption="Your photo", use_column_width=True)

    resized = image_source.resize((224, 224))
    img_array = np.array(resized).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("Looking closely..."):
        preds = model.predict(img_array, verbose=0)

    decoded = decode_predictions(preds, top=5)[0]
    top_label = decoded[0][1].replace("_", " ").title()
    top_confidence = decoded[0][2]

    with col2:
        st.subheader("My best guess:")
        st.markdown(f"### 🎯 {top_label}")
        st.caption(f"{top_confidence * 100:.1f}% confident")

    st.divider()
    st.subheader("Other possibilities")

    for _, label, confidence in decoded[1:]:
        readable_label = label.replace("_", " ").title()
        with st.container():
            st.markdown(f"**{readable_label}**  \n{confidence * 100:.1f}% confidence")
            st.progress(float(confidence))

    st.divider()
    if st.button("🔄 Try another photo"):
        st.session_state.uploader_key += 1
        st.rerun()

else:
    st.info("👆 Upload a photo or use your camera to get started")

# ---------- Footer ----------
st.divider()
st.caption("Built with a pretrained MobileNetV2 model · Powered by Streamlit")
