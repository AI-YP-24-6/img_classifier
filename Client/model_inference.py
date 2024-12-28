import streamlit as st
from PIL import Image
import requests

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
    uploaded_image = st.file_uploader("Загрузите изображение", type=['jpeg', 'png', 'jpg'])
    if uploaded_image is not None:
        with open('predict_image.jpg', 'rb') as image:
        # image = set_image_size(image)
            st.image(image, caption='Загруженное изображение', use_container_width=True)
        # запрос на предсказание
            response = requests.post(url_server + '/predict', files={"image": ("predict_image", image)})
            if response.status_code == 200:
                print("Файл успешно загружен!")
            else:
                print(f"Ошибка: {response.status_code} - {response.text}")
            
            name = "apple"
            probability = 0.91 * 100
            if st.toggle("Показать вероятность"):
                st.markdown(f':green-background[**Я думаю это {name} c вероятность {probability} %**]')
            else:
                st.markdown(f':green-background[**Я думаю это {name}**]')
            
        
    
    