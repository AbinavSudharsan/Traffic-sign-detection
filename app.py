import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile

model = YOLO("runs/detect/train/weights/best.pt")

st.title("Traffic Sign Detection App")

tab1, tab2 = st.tabs(["Image", "Video"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = model.predict(image)
        res_plotted = results[0].plot()
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        st.image(res_plotted, caption="Detection Result", use_container_width=True)

with tab2:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        st.info("Processing video (this may take a while)...")
        model.predict(tfile.name, save=True)

        st.video(tfile.name)

