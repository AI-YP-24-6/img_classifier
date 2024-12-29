import json
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image

from backend.app.api.models import DatasetInfo

CLASS_DICT = {}


def check_uploaded_file(upload_file: Any) -> bool:
    is_correct_format = True
    correct_format = (".jpg", ".jpeg")
    with zipfile.ZipFile(upload_file, "r") as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(correct_format):
                class_name = file_info.filename.split("/")[0]
                if class_name not in CLASS_DICT:
                    CLASS_DICT[class_name] = []
                CLASS_DICT[class_name].append(file_info.filename)
                continue
            elif not file_info.is_dir():
                st.error(f"Файл {file_info.filename} имеет неправильный формат или находится вне папки классов.")
                is_correct_format = False
    return is_correct_format


def display_images(uploaded_file):
    st.checkbox("reset")
    st.subheader("Примеры изображений по классам")
    for class_name, images in CLASS_DICT.items():
        st.write(f"**Класс: {class_name}**")
        images_to_display = images[:3]
        num_images = len(images_to_display)
        for i in range(0, num_images, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < num_images:
                    with zipfile.ZipFile(uploaded_file, "r") as z:
                        with z.open(images_to_display[i + j]) as img_file:
                            img = Image.open(img_file)
                            cols[j].image(img, caption=images_to_display[i + j])


def bar(classes, counts):
    plt.figure(figsize=(35, 20))
    plt.bar(classes, counts, color="#008080")
    y_pos = np.arange(len(classes))
    plt.xticks(y_pos, classes, rotation=90, ha="right", fontsize=30)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("Количество изображений", fontsize=30)
    st.pyplot(plt)
    plt.close()


def eda_page(url_server):
    st.header("EDA для датасета изображений")

    st.subheader("Загрузка данных")
    st.info("пожалуйста, убедитесь, что ваш датасет соответствует следующим требованиям:")
    st.markdown(
        """
    - **Формат изображений**: JPEG.
    - **Аннотации**: метки классов должны быть представлены в отдельной структуре папок.
    """
    )

    uploaded_file = st.file_uploader("Выберите файл с датасетом", type=["zip"])
    if uploaded_file is not None:
        if check_uploaded_file(uploaded_file):
            st.session_state.uploaded_file = uploaded_file
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(url_server + "/load_dataset", files=files)
            if response.status_code == 201:
                st.success("Датасет успешно загружен на сервер")
                print(response.text)
                try:
                    response_data = json.loads(response.text)
                    dataset_info = DatasetInfo(**response_data)
                    st.write("**Классы:**", dataset_info.classes)
                except Exception as e:
                    st.error(f"Ошибка при парсинге ответа: {e}")
            else:
                st.error(f"Произошла ошибка: {response.text}")

            st.subheader("Основные статистики:")
            st.markdown(
                """
            - **Средний размер изображений**: (100, 100)
            - **Средние значения по каналам RGB**: (0.3, 0.4, 0.5)).
            - **Средние отклонения по каналам RGB**: (0.3, 0.4, 0.5)).
            """
            )
            st.write("**Дубликаты**:", dataset_info.duplicates)
            st.subheader("График распределения изображений по классам:")
            bar(CLASS_DICT.keys(), len(CLASS_DICT.values()))
            display_images(uploaded_file)

        # response = -1
        # bar(response[0], response[1])
        # размер изображений, баланс классов, есть ли дубликаты, Средние значения и отклонения по каналам (R, G, B), визуализация нескольких изображений с каждого класса
        # график распределения изображений по классам, выявление аномалий
