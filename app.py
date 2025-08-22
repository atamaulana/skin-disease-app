import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown

# === Google Drive TFLite Model URL ===
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1SaISTM6nlOHtbhf3D4l1YbKrFr2oxzY2"
MODEL_PATH = "skin_disease_model.tflite"

# === Download model jika belum ada ===
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Label penyakit (urutan sesuai training) ===
CLASS_NAMES = [
    "Carcinoma",
    "Eczema",
    "Keratosis",
    "Milia",
    "Rosacea",
    "Acne",
    "Oily"
]

# === Preprocessing function ===
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
    return img_array

# === Prediction function ===
def predict(image: Image.Image):
    img_array = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

# === Streamlit UI ===
st.set_page_config(page_title="Skin Disease Classification", layout="wide")

st.title("🩺 Deteksi dini masalah kulit wajah")
st.write("Unggah Foto untuk Mendeteksi Penyakit Kulit pada Wajah.")

uploaded_file = st.file_uploader("📤 Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.info("🔎 Processing image...")

        # Prediksi
        preds = predict(image)
        top_idx = np.argmax(preds)
        confidence = preds[top_idx] * 100

        st.subheader(f"✅ Prediction: {CLASS_NAMES[top_idx]}")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Chart probabilitas
        fig, ax = plt.subplots()
        ax.barh(CLASS_NAMES, preds, color="skyblue")
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
