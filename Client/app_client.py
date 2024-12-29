import streamlit as st
from streamlit_option_menu import option_menu

from eda_page import eda_page
from model_inference import model_inference
from model_training_page import model_training_page

if "url_server" not in st.session_state:
    st.session_state.url_server = "http://127.0.0.1:8081/api/v1/models"

st.title("Сервис обучения моделей для классификации фруктов и овощей")

with st.sidebar:
    selected = option_menu("Разделы", ["EDA", "Обучение модели", "Инференс"])

if selected == "EDA":
    eda_page(st.session_state.url_server)
elif selected == "Обучение модели":
    model_training_page(st.session_state.url_server)
elif selected == "Инференс":
    model_inference(st.session_state.url_server)
