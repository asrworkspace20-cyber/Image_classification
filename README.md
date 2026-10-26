# 🖼️ AI-Powered Image Classifier

An interactive image classification web application built using **Python, Streamlit, TensorFlow, and MobileNetV2**. The application allows users to upload an image or capture a photo using their camera and predicts the objects present in the image.

## 🚀 Features

- 📁 Upload images in JPG, JPEG, and PNG formats.
- 📷 Capture photos directly using your camera.
- 🧠 Image classification using the pretrained MobileNetV2 deep learning model.
- 🎯 Display the most likely prediction with a confidence score.
- 📊 Show the top 5 predictions with confidence percentages.
- 🎨 Simple, interactive user interface built with Streamlit.
- ⚡ Model caching to reduce repeated loading time.
- 🔄 Easily classify another image using the reset option.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Streamlit** — Interactive web application framework
- **TensorFlow / Keras** — Deep learning model and inference
- **MobileNetV2** — Pretrained image classification model
- **NumPy** — Numerical operations and image array processing
- **Pillow (PIL)** — Image loading and processing

## 🧠 Model Information

This project uses MobileNetV2 pretrained on the ImageNet dataset.

- **Model:** MobileNetV2
- **Weights:** ImageNet
- **Input size:** 224 × 224 pixels
- **Output:** Top 5 predicted ImageNet classes with confidence scores
- **Categories:** 1,000 ImageNet classes

The model is used for inference, so this application does not train a new model.

## 📂 Project Structure

```text
ai-image-classifier/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-image-classifier.git
```

### 2. Navigate to the project directory

```bash
cd ai-image-classifier
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\\Scripts\\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

**Note:** On the first run, TensorFlow may download the pretrained ImageNet weights. An internet connection may be required.

## 📦 Requirements

Create a `requirements.txt` file with the following packages:

```text
streamlit
numpy
Pillow
tensorflow
```

For reproducible deployments, use versions tested in your environment rather than relying on unpinned latest releases.

## 📸 How to Use

1. Launch the application.
2. Choose **Upload a photo** or **Use camera**.
3. Upload an image or capture a photo.
4. View the predicted object and confidence score.
5. Explore the other top predictions.
6. Select **Try another photo** to classify a new image.

## 🎯 Use Cases

- Learning and demonstrating image classification
- Exploring deep learning and transfer learning
- Understanding pretrained computer vision models
- Building interactive AI-powered web applications
- Experimenting with ImageNet object categories

## ⚠️ Limitations

- Predictions are limited to the ImageNet categories supported by the pretrained model.
- Confidence scores are model outputs and should not be interpreted as guaranteed probabilities of correctness.
- Performance may vary with image quality, lighting, background complexity, and unfamiliar objects.
- The model may misclassify images containing multiple objects or specialized items.

## 🔮 Future Improvements

- Add drag-and-drop support for more image formats.
- Display prediction results in a more visual dashboard.
- Add a prediction history feature.
- Deploy the application to Streamlit Community Cloud.
- Explore custom-trained models for specific use cases.

## 👨‍💻 Author

**Abiya S Regi**

Connect with me on [LinkedIn](www.linkedin.com/in/abiya-s-regi).

---

⭐ If you find this project useful, consider giving the repository a star!
