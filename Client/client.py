import streamlit as st

from eda_page import eda_page
from model_training_page import model_training_page
from model_inference import model_inference

URL_SERVER = "http://127.0.0.1:8000"

st.title("Сервис обучения моделей для классификации фруктов и овощей")

st.sidebar.title("Меню")
page = st.sidebar.selectbox("Что вам хочется:", ["EDA", "Обучение модели", "Инференс"])

if page == "EDA":
    eda_page()
elif page == "Обучение модели":
    model_training_page()
elif page == "Инференс":
    model_inference()

        
        