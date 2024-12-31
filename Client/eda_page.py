import json
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from loguru import logger
from PIL import Image

from Backend.app.api.models import DatasetInfo


def show_bar(classes, counts):
    """Функция для построения столбчатых диаграмм."""
    plt.figure(figsize=(35, 20))
    plt.bar(classes, counts, color="#008080")
    y_pos = np.arange(len(classes))
    plt.xticks(y_pos, classes, rotation=90, ha="right", fontsize=30)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("Количество изображений", fontsize=30)
    st.pyplot(plt)
    plt.close()


def show_images(url_server):
    """Функция для отображения примеров изображений с каждого класса."""
    st.subheader("Примеры изображений по классам")
    try:
        with st.spinner("Ожидаем загрузки изображений..."):
            response = requests.get(url_server + "dataset/samples")
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Загруженные изображения", use_container_width=True)
            logger.info("Изображения успешно загружены и отображены для клиента")

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP ошибка при получении изображений с датасета: {http_err}")
        st.error("Ошибка получения изображений с датасета: Проверьте сервер.")

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Ошибка сети при получении изображений с датасета: {req_err}")
        st.error("Ошибка получения изображений с датасета: Проверьте соединение.")

    except OSError as io_err:
        logger.error(f"Ошибка обработки изображения: {io_err}")
        st.error("Ошибка обработки изображения: Не удалось открыть изображение.")


def show_bar_std_mean_rgb(rgb_df, cls):
    """Функция для отображения графика отклонений по каналам RGB для конкретного класса."""
    rows = rgb_df[rgb_df["class"] == cls].values
    mean_r = np.mean(rows[:, 2])
    mean_g = np.mean(rows[:, 3])
    mean_b = np.mean(rows[:, 4])
    std_r = np.std(rows[:, 5])
    std_g = np.std(rows[:, 6])
    std_b = np.std(rows[:, 7])

    means = [mean_r, mean_g, mean_b]
    stds = [std_r, std_g, std_b]
    channels = ["R", "G", "B"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=channels,
            y=means,
            error_y={"type": "data", "array": stds, "visible": True},
            marker={"color": ["red", "green", "blue"]},
            opacity=0.7,
        )
    )
    fig.update_layout(xaxis_title="Каналы", yaxis_title="Значение пикселей", width=800, height=400)

    st.plotly_chart(fig)
    logger.info(f"Отображение графика отклонений по каналам RGB для класса {cls}")


def show_eda(url_server):
    """Функция для отображения основных статистик датасета."""
    try:
        response = requests.get(url_server + "dataset/info")
        response_data = json.loads(response.text)
        dataset_info = DatasetInfo(**response_data)
        st.subheader("Основные статистики:")
        size_df = pd.DataFrame(dataset_info.sizes.rows, columns=dataset_info.sizes.columns)
        st.write(
            f"**Средний размер изображений**: "
            f"ширина: {round(size_df['width'].mean(), 0)}, "
            f"высота: {round(size_df['height'].mean(), 0)}"
        )
        logger.info("Вывод основных статистик для датасета")

        st.subheader("График распределения изображений по классам:")
        show_bar(dataset_info.classes.keys(), (dataset_info.classes.values()))
        logger.info("Отображение графика распределения изображений по классам")

        if dataset_info.duplicates is not None:
            st.subheader("График распределения дубликатов по классам:")
            show_bar(dataset_info.duplicates.keys(), dataset_info.duplicates.values())
            logger.info("Отображение графика распределения дубликатов по классам")
        else:
            st.write("**Дубликатов нет**")

        st.subheader("Среднее значение и стандартное отклонение по каналам (R, G, B)")
        rgb_df = pd.DataFrame(dataset_info.colors.rows, columns=dataset_info.colors.columns)
        classes = rgb_df["class"].unique()
        cls = st.selectbox("Выберите класс", classes)
        show_bar_std_mean_rgb(rgb_df, cls)

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP ошибка при получении EDA данных: {http_err}")
        st.error("Ошибка получения EDA данных: Проверьте сервер.")

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Ошибка сети при получении EDA данных: {req_err}")
        st.error("Ошибка получения EDA данных: Проверьте соединение.")

    except json.JSONDecodeError as json_err:
        logger.error(f"Ошибка декодирования JSON данных: {json_err}")
        st.error("Ошибка получения EDA данных: Неверный формат данных.")


def eda_page(url_server):
    """Функция для заполнения страницы с EDA."""
    st.header("EDA для датасета изображений")
    st.subheader("Загрузка данных")
    st.info("Пожалуйста, убедитесь, что ваш датасет соответствует следующим требованиям:")
    st.markdown(
        """
        - **Формат изображений**: JPEG.
        - **Аннотации**: метки классов должны быть представлены в отдельной структуре папок.
        """
    )
    uploaded_file = st.file_uploader("Выберите файл с датасетом", type=["zip"])
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("Ожидаем загрузки датасета на сервер..."):
            response = requests.post(url_server + "dataset/load", files=files)
            if response.status_code == 201:
                st.session_state.uploaded_file = uploaded_file
                st.success(f"Новый датасет {uploaded_file.name} успешно загружен на сервер")
                logger.info("Датасет успешно загружен на сервер")
                show_eda(url_server)
                show_images(url_server)
            else:
                logger.error(f"Произошла ошибка: {response.text}")
                st.error(f"Произошла ошибка: {response.text}")
    elif "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
        logger.info(f"На сервере уже есть датасет {st.session_state.uploaded_file.name}")
        st.subheader(f"**Датасет:** {st.session_state.uploaded_file.name}")
        show_eda(url_server)
        show_images(url_server)
