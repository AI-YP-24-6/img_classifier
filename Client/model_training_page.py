import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

from backend.app.api.models import FitRequest, ModelInfo


def plt_learning_curve(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes, train_scores_mean, marker="o", label="Тренировочная оценка", color="blue")
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.2
    )

    plt.plot(train_sizes, test_scores_mean, marker="o", label="Тестовая оценка", color="green")
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="green", alpha=0.2
    )

    plt.title("Кривые обучения")
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("Оценка")
    plt.xticks(train_sizes)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    plt.close()


def model_training_page(url_server):
    st.header("Создание новой модели SVC и выбор гиперпараметров")

    name_model = st.text_input("Введите название модели")
    param_c = st.slider("Выберите параметр регуляризации:", 0.1, 30.0, 0.1)
    kernel = st.selectbox("Выберите ядро:", ["linear", "poly", "rbf", "sigmoid", "precomputed"])
    probability = st.toggle("Включить оценку вероятности")
    learning_curve = st.toggle("Включить learning_curve")

    fit_request_data = FitRequest(
        config={"svc__C": param_c, "svc__kernel": kernel, "svc__probability": probability},
        with_learning_curve=learning_curve,
    )

    fit_json = fit_request_data.model_dump()
    if st.button(f":red[**Начать обучение модели**]"):
        with st.spinner("Обучение модели..."):
            response = requests.post(url_server + "/fit", json=fit_json)

            if response.status_code == 201:
                st.success("Модель успешно обучена!")
                try:
                    response_data = json.loads(response.text)
                    model_info = ModelInfo(**response_data)
                    st.subheader("Информация о модели")
                    st.markdown(
                        f"""
                        - **Название модели:** {model_info.id}
                        - **Параметр C =** {model_info.hyperparameters['svc__C']}
                        - **Ядро:** {model_info.hyperparameters['svc__kernel']}
                        - **Оценка вероятности:** {model_info.hyperparameters['svc__probability']}
                         """
                    )

                    st.subheader("Поученные кривые обучения")
                    learning_curve = model_info.learning_curve
                    print(f"Sizes = {learning_curve.train_sizes}")
                    plt_learning_curve(
                        learning_curve.train_sizes, learning_curve.train_scores, learning_curve.test_scores
                    )

                except Exception as e:
                    st.error(f"Ошибка при парсинге ответа: {e}")
            else:
                st.error(f"Произошла ошибка: {response.text}")
