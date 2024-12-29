import streamlit as st

from Client.model_inference import model_inference
from Client.model_training_page import model_training_page
from eda_page import eda_page

if "url_server" not in st.session_state:
    st.session_state.url_server = "http://127.0.0.1:8000/api/v1/models"

st.title("Сервис обучения моделей для классификации фруктов и овощей")

st.sidebar.title("Меню")
page = st.sidebar.selectbox("Что вам хочется:", ["EDA", "Обучение модели", "Инференс"])

if page == "EDA":
    eda_page(st.session_state.url_server)
elif page == "Обучение модели":
    model_training_page(st.session_state.url_server)
elif page == "Инференс":
    model_inference(st.session_state.url_server)
