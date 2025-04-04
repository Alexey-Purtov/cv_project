import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
from zipfile import ZipFile

def forest_app():
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('forest/best_model.h5')  # Updated path

    model = load_model()

    def process_single_image(image_file):
        image = Image.open(image_file)
        img_array = np.array(image)
        original_img = img_array.copy()
        img_processed = cv2.resize(img_array, (128, 128))
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR) / 255.0
        img_input = np.expand_dims(img_processed, axis=0)
        prediction = model.predict(img_input)
        pred_mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255
        return original_img, pred_mask

    def create_zip(files):
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w') as zip_file:
            for name, data in files.items():
                zip_file.writestr(name, data)
        zip_buffer.seek(0)
        return zip_buffer

    st.title("🌲 Пакетная сегментация лесного покрова")
    st.write("Загрузите несколько спутниковых изображений для сегментации")

    uploaded_files = st.file_uploader(
        "Выберите изображения...", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        zip_files = {}
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                with st.spinner(f'Обработка изображения {i+1}/{len(uploaded_files)}...'):
                    original_img, pred_mask = process_single_image(uploaded_file)
                    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                    colored_mask[pred_mask > 0] = [34, 139, 34]
                    overlay = cv2.addWeighted(
                        cv2.resize(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), (128, 128)), 
                        0.7, 
                        colored_mask, 
                        0.3, 
                        0
                    )
                    results.append({
                        'original': original_img,
                        'mask': pred_mask,
                        'overlay': overlay,
                        'name': uploaded_file.name
                    })
                    img_byte_arr = io.BytesIO()
                    Image.fromarray(pred_mask).save(img_byte_arr, format='PNG')
                    zip_files[f"mask_{uploaded_file.name}"] = img_byte_arr.getvalue()
            except Exception as e:
                st.error(f"Ошибка обработки файла {uploaded_file.name}: {str(e)}")

        if results:
            st.success(f"Успешно обработано {len(results)} изображений")
            cols_per_row = 3
            preview_size = 300 if len(results) <= 5 else 200

            for i, result in enumerate(results):
                if i % cols_per_row == 0:
                    cols = st.columns(cols_per_row)
                with cols[i % cols_per_row]:
                    st.image(result['original'], 
                            caption=f"Оригинал: {result['name']}", 
                            width=preview_size)
                    st.image(result['mask'], 
                            caption=f"Маска: {result['name']}", 
                            width=preview_size, 
                            clamp=True)
                    st.image(result['overlay'], 
                            caption=f"Наложение: {result['name']}", 
                            width=preview_size)

            if zip_files:
                zip_buffer = create_zip(zip_files)
                st.download_button(
                    label="📥 Скачать все маски (ZIP)",
                    data=zip_buffer,
                    file_name="forest_masks.zip",
                    mime="application/zip"
                )