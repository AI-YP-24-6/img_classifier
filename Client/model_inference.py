import json

import requests
import streamlit as st
from loguru import logger

from Backend.app.api.models import LoadRequest, PredictionResponse, ProbabilityResponse

def make_prediction(url_server, files, use_probability):
    endpoint = "models/predict_proba" if use_probability else "models/predict"
    response = requests.post(url_server + endpoint, files=files)

    if response.status_code == 200:
        response_data = json.loads(response.text)
        return response_data
    else:
        print(f"Ошибка: {response.status_code} - {response.text}")
        logger.exception("Ошибка на сервере во время предсказания")
        raise Exception(status_code=response.status_code)

def model_inference(url_server):
    st.header("Инференс с использованием обученной модели")
    if "model_info_list" in st.session_state:
        model_info_list = st.session_state.model_info_list
        model_names = [model.name for model in model_info_list]
        selected_model_name = st.selectbox("Выберите модель", model_names)
        selected_model_info = next((model for model in model_info_list if model.name == selected_model_name), None)
        with st.spinner("Загрузка модели для предсказания..."):
            logger.info(f"Началась загрузка модели {selected_model_name} для предсказания")
            load_model = LoadRequest(id=selected_model_info.id)
            load_json = load_model.model_dump()
            response = requests.post(url_server + "models/load", json=load_json)
            if response.status_code == 200:
                st.success(f"Модель {selected_model_name} успешно подготовлена для предсказания")
                logger.info(f"Модель {selected_model_name} успешно подготовлена для предсказания")
            else:
                logger.error(f"Ошибка получения ответа от сервера {response.status_code}")
            
    else:
        st.warning("Нет обученных моделей")
        logger.info("Нет обученных моделей")

    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpeg", "png", "jpg"])
    if uploaded_image is not None:
        logger.info("Изображение для предсказания успешно загружено")
        st.image(uploaded_image, caption="Загруженное изображение", use_container_width=True)

        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        if selected_model_info.id == "baseline":
            response_data = make_prediction(url_server, files, False)
        else:
            response_data = make_prediction(url_server, files, selected_model_info.hyperparameters["svc__probability"])

        if response_data:
            if selected_model_info.id != "baseline" and selected_model_info.hyperparameters["svc__probability"]:
                prediction_info = ProbabilityResponse(**response_data)
                st.markdown(
                    f""":green-background[**Я думаю это {prediction_info.prediction}
                            с вероятностью {round(prediction_info.probability,2)*100} %**]"""
                )
                logger.info("Предсказание с вероятность выполнено успешно")
            else:
                prediction_info = PredictionResponse(**response_data)
                st.markdown(f":green-background[**Я думаю это {prediction_info.prediction}**]")
                logger.info("Предсказание без вероятности выполнено успешно")
            
