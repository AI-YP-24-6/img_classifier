import streamlit as st
import matplotlib.pyplot as plt
from typing import Any
import zipfile
from PIL import Image
import numpy as np
import requests
from backend.app.api.models import DatasetInfo
import json
from io import BytesIO
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def bar(classes, counts):
    plt.figure(figsize=(35, 20))
    plt.bar(classes, counts, color="#008080")
    y_pos = np.arange(len(classes))
    plt.xticks(y_pos, classes, rotation=90, ha="right", fontsize=30)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("Количество изображений", fontsize=30)
    st.pyplot(plt)
    plt.close()
    
def show_images(url_server):
    st.subheader("Примеры изображений по классам")
    try:
        with st.spinner("Ожидаем загрузки изображений..."):
            response = requests.get(url_server + '/dataset_samples')
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Загруженные изображения", use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка получение изображений с датасета {e}")

def show_bar_std_mean_rgb(rgb_df, cls):
    rows = rgb_df[rgb_df['class'] == cls].values
    mean_r = np.mean(rows[:, 2])
    mean_g = np.mean(rows[:, 3])
    mean_b = np.mean(rows[:, 4])
    std_r = np.std(rows[:, 5])
    std_g = np.std(rows[:, 6])
    std_b = np.std(rows[:, 7])

    means = [mean_r, mean_g, mean_b]
    stds = [std_r, std_g, std_b]
    channels = ['R', 'G', 'B']

    # Создание графика
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels,
        y=means,
        error_y={'type': 'data', 'array': stds, 'visible': True},
        marker={'color': ['red', 'green', 'blue']},
        opacity=0.7
    ))
    fig.update_layout(
        xaxis_title='Каналы',
        yaxis_title='Значение пикселей',
        width=800,
        height=400
    )
    
    st.plotly_chart(fig)
    
def show_eda(url_server):
    try:
        response = requests.get(url_server + '/dataset_info')
        response_data = json.loads(response.text)
        dataset_info = DatasetInfo(**response_data)
        st.subheader("Основные статистики:")
        size_df = pd.DataFrame(dataset_info.sizes, columns=['class', 'name', 'width', 'height'])
        st.write(f'**Средний размер изображений**: ширина: {round(size_df['width'].mean(),0)}, высота: {round(size_df['height'].mean(),0)}')
        
        st.subheader("График распределения изображений по классам:")
        bar(dataset_info.classes.keys(), (dataset_info.classes.values()))
        
        if dataset_info.duplicates is not None:
            st.subheader("График распределения дубликатов по классам:")
            bar(dataset_info.duplicates.keys(), dataset_info.duplicates.values())
        else:
            st.write("**Дубликатов нет**")
        
        st.subheader("Среднее значение и стандартное отклонение по каналам (R, G, B)")
        rgb_df = pd.DataFrame(dataset_info.colors, columns=['class', 'name', 'mean_R', 'mean_G', 'mean_B', 'std_R', 'std_G', 'std_B'])
        classes = rgb_df['class'].unique()
        cls = st.selectbox("Выберите класс", classes)
        show_bar_std_mean_rgb(rgb_df, cls)
        
    except Exception as e:
        st.error(f"Ошибка получение EDA данных, загрузите датасет на сервер")
      
def eda_page(url_server):
    st.header("EDA для датасета изображений")
    st.subheader("Загрузка данных")
    st.info("Пожалуйста, убедитесь, что ваш датасет соответствует следующим требованиям:")
    st.markdown("""
        - **Формат изображений**: JPEG.
        - **Аннотации**: метки классов должны быть представлены в отдельной структуре папок.
        """)
    uploaded_file = st.file_uploader("Выберите файл с датасетом", type=["zip"])
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("Ожидаем загрузки датасета на сервер..."):
            response = requests.post(url_server + '/load_dataset', files=files)
            if response.status_code == 201:
                st.session_state.uploaded_file = uploaded_file
                st.success("Датасет успешно загружен на сервер")
                show_eda(url_server)
                show_images(url_server)
            else:
                st.error(f"Произошла ошибка: {response.text}")
    elif "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
        show_eda(url_server)
        show_images(url_server)
