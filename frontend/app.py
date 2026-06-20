import os

# macOS TensorFlow threading workaround
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():

    import tensorflow as tf

    model = tf.keras.models.load_model(
        "neu_defect_classifier.keras"
    )

    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed:\n{e}")
    st.stop()

# ----------------------------
# Class Names
# ----------------------------

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ----------------------------
# Preprocessing
# ----------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ----------------------------
# UI
# ----------------------------

st.title("Steel Surface Defect Detection")

st.write(
    "Upload a steel surface image and the model will classify the defect."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

# ----------------------------
# Prediction
# ----------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    processed = preprocess_image(
        image
    )

    prediction = model.predict(
        processed,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    predicted_class = class_names[
        predicted_index
    ]

    st.subheader("Prediction")

    st.success(
        f"Defect: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    st.subheader("Class Probabilities")

    for cls, prob in zip(
        class_names,
        prediction[0]
    ):

        st.write(
            f"{cls}: {prob:.4f}"
        )

    st.subheader("Raw Output")

    st.write(
        prediction
    )