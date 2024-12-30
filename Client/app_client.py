import streamlit as st
from eda_page import eda_page
from loguru import logger
from model_inference import model_inference
from model_training_page import model_training_page
from streamlit_option_menu import option_menu

LOG_FOLDER = "logs"


def configure_logging():
    logger.remove()
    os.makedirs(LOG_FOLDER, exist_ok=True)
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|<level>{level}</level>| {message}",
    )
    logger.add(
        os.path.join(LOG_FOLDER, "client.log"),
        colorize=True,
        format="{time} | {level} | {message}",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
    )

    class InterceptHandler:
        def write(self, message):
            if message.strip():
                logger.info(message.strip())

    sys.stderr = InterceptHandler()


if "url_server" not in st.session_state:
    st.session_state.url_server = "http://127.0.0.1:8081/api/v1"
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
