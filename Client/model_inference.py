import streamlit as st
from PIL import Image
import requests
from backend.app.api.models import LoadRequest, PredictionResponse
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
            st.success("Модель успешно подготовлена для предсказания")  
    else:
        st.warning("Нет обученных моделей")
    
    uploaded_image = st.file_uploader("Загрузите изображение", type=['jpeg', 'png', 'jpg'])
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Загруженное изображение', use_container_width=True)
        
        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        response = requests.post(url_server + '/predict', files=files)
        
        if response.status_code == 200:
            print("Файл успешно загружен!")
            response_data = json.loads(response.text)
            prediction_info = PredictionResponse(**response_data)
            st.markdown(f':green-background[**Я думаю это {prediction_info.prediction}**]')
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")
            
            # name = "apple"
            # probability = 0.91 * 100
            # if st.toggle("Показать вероятность"):
            #     st.markdown(f':green-background[**Я думаю это {name} c вероятность {probability} %**]')
            # else:
            #     st.markdown(f':green-background[**Я думаю это {name}**]')
            
        
    
    