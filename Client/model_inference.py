import json
from io import BytesIO

import requests
import streamlit as st
from loguru import logger

from Backend.app.api.models import LoadRequest, ModelInfo, PredictionResponse, ProbabilityResponse


def make_prediction(url_server: str, files: BytesIO, use_probability: bool) -> dict:
    """Функция для получения предсказания на обученной модели."""
    try:
        endpoint = "models/predict_proba" if use_probability else "models/predict"
        response = requests.post(f"{url_server}{endpoint}", files=files)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP ошибка во время предсказания: {http_err}")
        logger.error(f"HTTP ошибка во время предсказания: {http_err}")
        return None

    except requests.exceptions.RequestException as req_err:
        st.error(f"Сетевая ошибка во время предсказания: {req_err}")
        logger.error(f"Сетевая ошибка во время предсказания: {req_err}")
        return None

    except json.JSONDecodeError as json_err:
        st.error(f"Ошибка декодирования JSON: {json_err}")
        logger.error(f"Ошибка декодирования JSON: {json_err}")
        return None


def download_trained_model(url_server: str, selected_model_info: ModelInfo) -> bool:
    """Функция загрузки обученной модели на сервер"""
    with st.spinner("Загрузка модели для предсказания..."):
        try:
            logger.info(f"Началась загрузка модели {selected_model_info.name} для предсказания")

            load_model = LoadRequest(id=selected_model_info.id)
            load_json = load_model.model_dump()

            requests.post(f"{url_server}models/load", json=load_json)
            st.success(f"Модель {selected_model_info.name} успешно подготовлена для предсказания")
            logger.info(f"Модель {selected_model_info.name} успешно подготовлена для предсказания")
            return True

        except requests.exceptions.RequestException as e:
            st.error("Произошла ошибка при попытке загрузить модель. Проверьте соединение с сервером.")
            logger.exception(f"Ошибка получения ответа от сервера: {e}")
            return False


def model_inference(url_server: str):
    """Функция для получения предсказания на обученной модели."""
    st.header("Инференс с использованием обученной модели")

    if "model_info_list" in st.session_state:
        model_info_list = st.session_state.model_info_list
        model_names = [model.name for model in model_info_list]
        selected_model_name = st.selectbox("Выберите модель", model_names)
        selected_model_info = next((model for model in model_info_list if model.name == selected_model_name), None)

        if download_trained_model(url_server, selected_model_info):
            uploaded_image = st.file_uploader("Загрузите изображение", type=["jpeg", "png", "jpg"])
            if uploaded_image is not None:
                logger.info("Изображение для предсказания успешно загружено")
                st.image(uploaded_image, caption="Загруженное изображение", use_container_width=True)

                files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                if selected_model_info.id == "baseline":
                    response_data = make_prediction(url_server, files, False)
                else:
                    response_data = make_prediction(
                        url_server, files, selected_model_info.hyperparameters["svc__probability"]
                    )

                if response_data:
                    if selected_model_info.id != "baseline" and selected_model_info.hyperparameters["svc__probability"]:
                        prediction_info = ProbabilityResponse(**response_data)
                        st.markdown(
                            f""":green-background[**Я думаю это {prediction_info.prediction}
                            с вероятностью {round(prediction_info.probability, 2) * 100} %**]"""
                        )
                        logger.info("Предсказание с вероятность выполнено успешно")
                    else:
                        prediction_info = PredictionResponse(**response_data)
                        st.markdown(f":green-background[**Я думаю это {prediction_info.prediction}**]")
                        logger.info("Предсказание без вероятности выполнено успешно")
