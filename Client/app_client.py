import os

import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from streamlit_option_menu import option_menu

from Client.eda_page import eda_page
from Client.model_inference import model_inference
from Client.model_training_page import model_training_page
from Tools.logger_config import configure_client_logging

# Load the .env file
load_dotenv()

configure_client_logging("../logs/")


if "url_server" not in st.session_state:
    st.session_state.url_server = os.getenv("BACKEND_URL", "http://127.0.0.1:54545/api/v1/")
logger.info("Запуск клиента")
st.title("Сервис обучения моделей для классификации фруктов и овощей")

with st.sidebar:
    selected = option_menu("Разделы", ["EDA", "Обучение модели", "Инференс"])

if selected == "EDA":
    logger.info("Выбрана страница для загрузки датасета и EDA")
    eda_page(st.session_state.url_server)
elif selected == "Обучение модели":
    logger.info("Выбрана страница для обучения модели")
    model_training_page(st.session_state.url_server)
elif selected == "Инференс":
    logger.info("Выбрана страница для инференса")
    model_inference(st.session_state.url_server)

logger.info("Завершение работы клиента")
