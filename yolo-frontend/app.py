import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Steel Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🔍 Steel Surface Defect Detection")

st.write(
    "Upload a steel surface image to detect defects using YOLOv8."
)

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # -----------------------------------------
    # SAVE TEMP FILE
    # -----------------------------------------

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    image.save(temp.name)

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    start = time.time()

    results = model.predict(
        source=temp.name,
        conf=0.25,
        verbose=False
    )

    inference_time = (
        time.time() - start
    ) * 1000

    result = results[0]

    # -----------------------------------------
    # DRAW RED BOXES
    # -----------------------------------------

    output = np.array(image).copy()

    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy.cpu().numpy()[0].astype(int)
        )

        cls = int(box.cls)

        conf = float(box.conf)

        label = f"{model.names[cls]} {conf:.2f}"

        # Red Bounding Box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),   # RGB Red
            3
        )

        # Text Size
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        # Filled Red Rectangle
        cv2.rectangle(
            output,
            (x1, y1 - th - 12),
            (x1 + tw + 10, y1),
            (255, 0, 0),
            -1
        )

        # White Text
        cv2.putText(
            output,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    with col2:

        st.subheader("Detection")

        st.image(
            output,
            use_container_width=True
        )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    st.divider()

    st.subheader("Detection Summary")

    c1, c2 = st.columns(2)

    c1.metric(
        "Detected Defects",
        len(result.boxes)
    )

    c2.metric(
        "Inference Time",
        f"{inference_time:.1f} ms"
    )

    # -----------------------------------------
    # DETECTION DETAILS
    # -----------------------------------------

    if len(result.boxes) == 0:

        st.success("No defects detected.")

    else:

        st.subheader("Detected Defects")

        for i, box in enumerate(result.boxes):

            cls = int(box.cls)

            conf = float(box.conf)

            coords = (
                box.xyxy.cpu()
                .numpy()[0]
                .astype(int)
                .tolist()
            )

            with st.expander(f"Defect {i+1}"):

                st.write(
                    f"**Class:** {model.names[cls]}"
                )

                st.write(
                    f"**Confidence:** {conf:.2%}"
                )

                st.write(
                    f"**Bounding Box:** {coords}"
                )