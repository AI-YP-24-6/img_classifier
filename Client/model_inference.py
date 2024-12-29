import streamlit as st
from PIL import Image
import requests
from backend.app.api.models import LoadRequest, PredictionResponse, ProbabilityResponse
import json

SIZE_IMG = (20, 20)
def set_image_size(img: Image) ->  Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    ratio = img.width / img.height
    
    if ratio > 1:
        new_width = SIZE_IMG[0]
        new_height = int(SIZE_IMG[0] / ratio)
    
    else:
        new_height = SIZE_IMG[1]
        new_width = int(SIZE_IMG[1] * ratio)
        
    return img.resize((new_width, new_height), Image.LANCZOS)

def make_prediction(url_server, files, use_probability):
    endpoint = '/predict_proba' if use_probability else '/predict'
    response = requests.post(url_server + endpoint, files=files)

    if response.status_code == 200:
        response_data = json.loads(response.text)
        return response_data
    else:
        print(f"Ошибка: {response.status_code} - {response.text}")
        return None


def model_inference(url_server):
    st.header("Инференс с использованием обученной модели")
    if 'model_info_list' in st.session_state:
        model_info_list = st.session_state.model_info_list
        model_names = [model.name for model in model_info_list]
        selected_model_name = st.selectbox("Выберите модель", model_names)
        selected_model_info = next((model for model in model_info_list if model.name == selected_model_name), None)
        with st.spinner("Загрузка модели для предсказания..."):
            load_model = LoadRequest(id=selected_model_info.id)
            load_json = load_model.model_dump()
            response = requests.post(url_server + "/load", json=load_json)      
            st.success(f"Модель {selected_model_name} успешно подготовлена для предсказания")  
    else:
        st.warning("Нет обученных моделей")
    
    uploaded_image = st.file_uploader("Загрузите изображение", type=['jpeg', 'png', 'jpg'])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Загруженное изображение', use_container_width=True)
        
        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        response_data = make_prediction(url_server, files, selected_model_info.hyperparameters['svc__probability'])

        if response_data:
            if selected_model_info.hyperparameters['svc__probability']:
                prediction_info = ProbabilityResponse(**response_data)
                st.markdown(f':green-background[**Я думаю это {prediction_info.prediction} с вероятностью {round(prediction_info.probability,2)*100} %**]')
            else:
                prediction_info = PredictionResponse(**response_data)
                st.markdown(f':green-background[**Я думаю это {prediction_info.prediction}**]')
            
        
    
    