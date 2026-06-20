import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "neu_defect_classifier.keras"
    )

model = load_model()

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

def preprocess_image(image):

    image = image.resize(
        (224,224)
    )

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

st.title(
    "Steel Surface Defect Detection"
)

uploaded_file = st.file_uploader(
    "Upload Steel Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image"
    )

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    processed = preprocess_image(
        image
    )

    prediction = model.predict(
        processed
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

st.write(
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