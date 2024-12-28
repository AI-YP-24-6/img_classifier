import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.svm import SVC
import numpy as np
import requests
import json
from backend.app.api.models import FitRequest, ModelInfo

def plt_learning_curve(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_scores_mean, marker='o', label='Тренировочная оценка', color='blue')
    plt.fill_between(train_sizes, 
                 train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, 
                 color='blue', alpha=0.2)
    
    plt.plot(train_sizes, test_scores_mean, marker='o', label='Тестовая оценка', color='green')
    plt.fill_between(train_sizes, 
                 test_scores_mean - test_scores_std, 
                 test_scores_mean + test_scores_std, 
                 color='green', alpha=0.2)
    
    plt.title("Кривые обучения")
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("f1-macro")
    plt.xticks(train_sizes)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    plt.close()

def show_model_statistics(model_info):
    st.subheader("Информация о модели")
    st.markdown(f"""
                - **Название модели:** {model_info.name}
                - **Параметр C =** {model_info.hyperparameters['svc__C']}
                - **Ядро:** {model_info.hyperparameters['svc__kernel']}
                - **Оценка вероятности:** {model_info.hyperparameters['svc__probability']}
                """)
    
    learning_curve = model_info.learning_curve
    if learning_curve is not None:
        st.subheader("Поученные кривые обучения")
        plt_learning_curve(learning_curve.train_sizes, learning_curve.train_scores, learning_curve.test_scores)
    
def model_training_page(url_server):
    st.subheader("Выбор существующих моделей")
    try:
        with st.spinner("Загрузка списка моделей..."):
            response = requests.get(url_server + '/list_models')
            model_data = response.json()
            if not model_data:
                st.warning("Список моделей пуст.")
            else:
                model_info_list = []
                for _, model_info in model_data.items():
                    model_info_list.append(ModelInfo(**model_info)) 
                 
                model_names = [model.name for model in model_info_list]
                selected_model_name = st.selectbox("Выберите уже обученную модель", model_names)
                selected_model_info = next((model for model in model_info_list if model.name == selected_model_name), None)
                show_model_statistics(selected_model_info)
            
    except Exception as e:
        st.error(f"На сервере ошибка {e}")
        
    st.subheader("Создание новой модели SVC и выбор гиперпараметров")
    
    name_model = st.text_input("Введите название модели")
    param_c = st.slider("Выберите параметр регуляризации:", 0.1, 30.0, 0.1)
    kernel = st.selectbox("Выберите ядро:", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    probability = st.toggle("Включить оценку вероятности")
    learning_curve = st.toggle("Включить learning_curve")
    
    fit_request_data = FitRequest(
        config={
        "svc__C": param_c,
        "svc__kernel": kernel,
        "svc__probability": probability
        },
        with_learning_curve = learning_curve,
        name = name_model
    )
    
    fit_json = fit_request_data.model_dump()
    if st.button(f':red[**Начать обучение модели**]'):
        with st.spinner("Обучение модели..."):
            response = requests.post(url_server + "/fit", json=fit_json)
        
            if response.status_code == 201:
                st.success("Модель успешно обучена!")
                try:
                    response_data = json.loads(response.text)
                    model_info = ModelInfo(**response_data)
                    show_model_statistics(model_info)
                except Exception as e:
                    st.error(f"Ошибка при парсинге ответа: {e}")
            else:
                st.error(f"Произошла ошибка: {response.text}")
    

    
    