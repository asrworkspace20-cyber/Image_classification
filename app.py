import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


@st.cache_resource
def load_model():
    # Loads once and stays cached across user sessions/refreshes
    return keras.models.load_model('cifar10_model.keras')


model = load_model()

st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Upload a photo and the model will guess which of these 10 categories it "
    "belongs to: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
)
st.caption(
    "Note: this model was trained on tiny 32x32 images, so it works best on simple, "
    "single-subject photos rather than busy real-world scenes."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Your upload", use_column_width=True)

    # Preprocess to match training: 32x32, normalized, 3 channels
    resized = img.resize((32, 32))
    img_array = np.array(resized).astype('float32') / 255.0

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Classifying..."):
        preds = model.predict(img_array, verbose=0)[0]

    # Sort by confidence, show top 3
    top_indices = np.argsort(preds)[::-1][:3]

    st.subheader("Predictions")
    for i in top_indices:
        st.write(f"**{class_names[i]}**")
        st.progress(float(preds[i]))
        st.caption(f"{preds[i] * 100:.1f}% confidence")
