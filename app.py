import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deepfake Tespit Sistemi",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Deepfake Tespit Sistemi")

st.write(
    "Bu uygulama, yüklenen bir görselin gerçek mi yoksa deepfake mi olduğunu "
    "ve hangi sınıfa ait olabileceğini tahmin eder."
)

st.warning(
    "Not: Model sonucu kesin doğruluk anlamına gelmez. "
    "Bu sonuç, eğitilen CNN modelinin olasılıksal tahminidir."
)

@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("deepfake_multiclass_cnn_final.keras")
    return model

model = load_cnn_model()

IMG_SIZE = 128

class_names = [
    "DALL-E",
    "DeepFaceLab",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "Midjourney",
    "NeuralTextures",
    "Real",
    "Stable Diffusion",
    "StyleGAN"
]

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

uploaded_file = st.file_uploader(
    "Bir görsel yükleyin",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Yüklenen Görsel",
        use_container_width=True
    )

    if st.button("Tahmin Et"):
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        class_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        predicted_class = class_names[class_index]

        st.subheader("Sonuç")

        if predicted_class == "Real":
            st.success(f"Tahmin: {predicted_class}")
            st.write("Model bu görseli **gerçek** olarak sınıflandırdı.")
        else:
            st.error(f"Tahmin: {predicted_class}")
            st.write("Model bu görseli **deepfake / yapay üretilmiş** olarak sınıflandırdı.")

        st.write(f"**Güven oranı:** %{confidence:.2f}")

        st.subheader("Sınıf Olasılıkları")

        for i, class_name in enumerate(class_names):
            probability = prediction[0][i] * 100
            st.write(f"**{class_name}:** %{probability:.2f}")
            st.progress(float(prediction[0][i]))