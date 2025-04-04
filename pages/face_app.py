import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import os
from ultralytics import YOLO
from blur import blur_faces  
import pandas as pd

def face_app():
    @st.cache_resource
    def load_model():
        return "face/face_model.pt"  # Updated path

    model = load_model()

    st.title("🟣 Анонимайзер лиц с помощью YOLO")
    option = st.radio("Выберите источник изображения:", ["Файлы", "Ссылки"])
    images = []

    if option == "Файлы":
        uploaded_files = st.file_uploader("Загрузите изображение(я)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            images.append(image)

    elif option == "Ссылки":
        urls = st.text_area("Вставьте прямые ссылки на изображения (по одной на строку)")
        for url in urls.splitlines():
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(image)
            except Exception as e:
                st.error(f"Не удалось загрузить изображение по ссылке: {url}\n{e}")

    if images:
        results = blur_faces(model, images)
        for orig, blurred in zip(images, results):
            st.image(orig, caption="Оригинал", use_container_width=True)
            st.image(blurred, caption="С результатом", use_container_width=True)

        st.subheader("📊 Информация о модели")
        csv_path = "face/results.csv"  # Updated path

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.markdown(f"**Число эпох:** {len(df)}")
            st.markdown(f"**Объём выборки:** {df['train/box_loss'].count()} (примерно)")

            precision_metrics = [col for col in df.columns if "precision" in col]
            recall_metrics = [col for col in df.columns if "recall" in col]
            map_metrics = [col for col in df.columns if "mAP" in col]
            loss_metrics = [col for col in df.columns if "loss" in col]

            if precision_metrics:
                st.subheader("🎯 Precision по эпохам")
                st.line_chart(df[precision_metrics])

            if recall_metrics:
                st.subheader("📥 Recall по эпохам")
                st.line_chart(df[recall_metrics])

            if map_metrics:
                st.subheader("🌍 mAP по эпохам")
                st.line_chart(df[map_metrics])

            if loss_metrics:
                st.subheader("📉 Loss по эпохам")
                st.line_chart(df[loss_metrics])

            for name in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
                path = os.path.join("face", name)
                if os.path.exists(path):
                    st.image(path, caption=name)
        else:
            st.warning("Файл results.csv не найден.")
