import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import sys

st.set_page_config(
    page_title="Deepfake Tespit Sistemi",
    page_icon="🔎",
    layout="wide"
)

MODEL_PATH = Path("deepfake_multiclass_cnn_no_preprocess.keras")
IMG_SIZE = 128

CLASS_NAMES = [
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

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .app-header {
        border-bottom: 1px solid #e6e8ec;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
    }
    .app-header h1 {
        font-size: 2.2rem;
        line-height: 1.15;
        margin-bottom: .35rem;
    }
    .app-header p {
        color: #5b6472;
        font-size: 1rem;
        margin: 0;
    }
    .result-panel {
        border: 1px solid #e6e8ec;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #ffffff;
    }
    .muted {
        color: #687386;
        font-size: .92rem;
    }
    div.stButton > button {
        width: 100%;
        height: 2.8rem;
        border-radius: 8px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Deepfake Tespit Sistemi</h1>
        <p>Yüklenen görseli eğitilmiş CNN modeliyle analiz eder ve en olası üretim sınıfını gösterir.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def get_startup_error():
    if sys.version_info >= (3, 13):
        return (
            "Uygulama Python 3.13 ile açılmış. Bu projede TensorFlow modeli Python 3.13 altında "
            "kararsız çalışıyor. Python 3.11 ile sanal ortam oluşturup tekrar başlatın."
        )

    if not MODEL_PATH.exists():
        return f"Model dosyası bulunamadı: {MODEL_PATH}"

    return None

@st.cache_resource
def load_cnn_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)

def prepare_model_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = ImageOps.fit(
        image,
        (IMG_SIZE, IMG_SIZE),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )

    return image

def preprocess_image(image):
    image = prepare_model_image(image)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def run_prediction(model, image):
    prediction = model.predict(preprocess_image(image), verbose=0)
    probabilities = np.asarray(prediction[0], dtype=float)

    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Model {probabilities.shape[0]} sınıf döndürdü, uygulamada {len(CLASS_NAMES)} sınıf tanımlı."
        )

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": confidence,
        "probabilities": probabilities,
    }

startup_error = get_startup_error()
model = None

with st.sidebar:
    st.subheader("Sistem Durumu")
    st.caption(f"Python: {sys.version.split()[0]}")
    st.caption(f"Model dosyası: {MODEL_PATH.name}")

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        st.success(f"Model bulundu ({size_mb:.1f} MB)")
    else:
        st.error("Model dosyası yok")

    st.divider()
    st.caption("Beklenen giriş")
    st.write(f"Merkezden kare kırpılmış {IMG_SIZE} x {IMG_SIZE} RGB görsel")
    st.caption("Sınıf sayısı")
    st.write(str(len(CLASS_NAMES)))

if not startup_error:
    try:
        with st.spinner("Model yükleniyor..."):
            model = load_cnn_model()
    except Exception as exc:
        startup_error = f"Model yüklenemedi: {exc}"

if startup_error:
    st.error("Model hazır değil.")
    st.warning(startup_error)
    st.code(
        "python3.11 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="bash",
    )
    st.stop()

left_col, right_col = st.columns([0.9, 1.1], gap="large")

with left_col:
    st.subheader("Görsel Yükle")
    uploaded_file = st.file_uploader(
        "JPG, PNG veya WEBP formatında bir görsel seçin",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as exc:
            st.error(f"Görsel okunamadı: {exc}")

    if image is None:
        st.info("Analiz için bir görsel yükleyin.")
    else:
        st.image(image, caption="Yüklenen görsel", use_container_width=True)
        st.caption(f"Orijinal boyut: {image.size[0]} x {image.size[1]} px")
        with st.expander("Modele gönderilen önizleme"):
            st.image(
                prepare_model_image(image),
                caption=f"{IMG_SIZE} x {IMG_SIZE} kare giriş",
                use_container_width=False,
            )

    analyze = st.button("Analiz Et", type="primary", disabled=image is None)

with right_col:
    st.subheader("Analiz Sonucu")

    if not analyze:
        st.markdown(
            '<div class="result-panel"><p class="muted">Sonuçlar burada görünecek.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Görsel analiz ediliyor..."):
                result = run_prediction(model, image)
        except Exception as exc:
            st.error(f"Tahmin yapılamadı: {exc}")
            st.stop()

        predicted_class = result["class_name"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        label = "Gerçek" if predicted_class == "Real" else "Yapay / Deepfake"
        if predicted_class == "Real":
            st.success(f"Tahmin: {predicted_class}")
        else:
            st.error(f"Tahmin: {predicted_class}")

        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Karar", label)
        metric_col_2.metric("Güven", f"%{confidence * 100:.2f}")

        st.caption(
            "Bu sonuç modelin olasılıksal tahminidir; kesin doğruluk veya adli tespit anlamına gelmez."
        )

        st.divider()
        st.subheader("Sınıf Olasılıkları")

        sorted_indices = np.argsort(probabilities)[::-1]
        for index in sorted_indices:
            probability = float(probabilities[index])
            st.write(f"**{CLASS_NAMES[index]}** · %{probability * 100:.2f}")
            st.progress(max(0.0, min(1.0, probability)))
