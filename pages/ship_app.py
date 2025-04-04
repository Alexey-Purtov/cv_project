import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

def ship_app():
    @st.cache_resource
    def load_model():
        return YOLO("ship/best.pt")  # Updated path

    model = load_model()

    st.title("Ship Detection with YOLO")
    st.write("Upload an image for analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(image, conf=0.5)
        annotated_image = results[0].plot(line_width=2)
        st.image(annotated_image, caption="Detection Results", use_container_width=True)
        detected_objects = len(results[0].boxes)
        st.subheader(f"Detected ships: {detected_objects}")
        
        if detected_objects > 0:
            confidences = [round(float(box.conf), 2) for box in results[0].boxes]
            st.write(f"Confidence scores: {confidences}")