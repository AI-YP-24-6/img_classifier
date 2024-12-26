import streamlit as st
from PIL import Image

def model_inference():
    st.header("Инференс с использованием обученной модели")
    uploaded_image = st.file_uploader("Загрузите изображение", type=['jpeg', 'png', 'jpg'])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # image = image.thumbnail(size=(10,10), resample = 3)
        st.image(image, caption='Загруженное изображение', use_container_width=True)
    
    