import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import streamlit as st
from loguru import logger

from Backend.app.api.models import FitRequest, ModelInfo


def plt_learning_curve(model_info_list: list[ModelInfo]) -> None:
    """Функция для отображения графика кривых обучения модели."""
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(model_info_list))

    for index, model in enumerate(model_info_list):

        train_scores_mean = np.mean(model.learning_curve.train_scores, axis=1)
        train_scores_std = np.std(model.learning_curve.train_scores, axis=1)

        test_scores_mean = np.mean(model.learning_curve.test_scores, axis=1)
        test_scores_std = np.std(model.learning_curve.test_scores, axis=1)

        plt.plot(
            model.learning_curve.train_sizes,
            train_scores_mean,
            marker="o",
            label=f"{model.name} - Тренировочная оценка",
            color=colors[index],
        )
        plt.fill_between(
            model.learning_curve.train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            color=colors[index],
            alpha=0.2,
        )

        plt.plot(
            model.learning_curve.train_sizes,
            test_scores_mean,
            marker="o",
            label=f"{model.name} - Тестовая оценка",
            color=colors[index],
            linestyle="--",
        )

        plt.fill_between(
            model.learning_curve.train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            color=colors[index],
            alpha=0.2,
        )

    plt.title("Кривые обучения")
    plt.xlabel("Размер обучающей выборки")
    plt.ylabel("f1-macro")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    plt.close()
    logger.info("Построен график кривых обучения модели")


def change_models_learning_curve() -> None:
    """Функция выбора моделей, для которых будет постоен график кривых обучения."""
    if "model_info_list" in st.session_state:
        st.subheader("Построение графиков кривых обучения моделей")
        model_info_list = st.session_state.model_info_list
        model_learning_curve = []
        valid_models = []
        for model_info in model_info_list:
            if model_info.learning_curve is not None:
                valid_models.append(model_info)
                if st.checkbox(model_info.name, value=False):
                    model_learning_curve.append(model_info)

        if model_learning_curve:
            plt_learning_curve(model_learning_curve)
        elif valid_models:
            st.warning("Выберите хотя бы одну модель для отображения.")
    else:
        st.error("Список моделей не найден в состоянии сессии.")
        logger.error("Список моделей не найден в состоянии сессии.")


def show_model_statistics(model_info: ModelInfo) -> None:
    """Функция для отображения информации об обученной модели."""

    st.subheader("Информация о модели")
    hyperparams_str = "".join([f"\n- **{key} =** {value}" for key, value in model_info.hyperparameters.items()])
    st.markdown(
        f"""
    **Название модели:** {model_info.name} <br>
    **Гиперпараметры:** {hyperparams_str}
    """,
        unsafe_allow_html=True,
    )

    logger.info("Для клиента отображена основная информация об обученной модели")


def delete_model(url_server: str, model_id: str) -> bool:
    """Функция для удаления обученной модели."""
    try:
        response = requests.delete(url_server + f"models/remove/{model_id}", timeout=90)
        if response.status_code == 200:
            return True

    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера.")
        logger.error("Превышено время ожидания ответа от сервера")

    return False


def delete_all_models(url_server: str) -> bool:
    """Функция для удаления всех обученных моделей."""
    try:
        response = requests.delete(url_server + "models/remove_all", timeout=90)
        if response.status_code == 200:
            return True

    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера.")
        logger.error("Превышено время ожидания ответа от сервера")

    return False


def get_models_list(url_server: str) -> list[ModelInfo] | None:
    """Функция для получения списка всех обученных моделей."""
    try:
        with st.spinner("Загрузка списка моделей..."):
            logger.info("Загрузка списка моделей с сервера")
            model_info_list = []
            response = requests.get(url_server + "models/list_models", timeout=90)
            model_data = response.json()
            if not model_data:
                st.warning("Список моделей пуст.")
                logger.info("Список моделей пуст")
            else:
                for _, model_info in model_data.items():
                    model_info_list.append(ModelInfo(**model_info))

                st.session_state.model_info_list = model_info_list
                logger.info(f"Получен список обученных моделей {model_info_list}")
            return model_info_list

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP ошибка при получении списка моделей: {http_err}")
        st.error("Ошибка получения списка моделей: Проверьте сервер.")
        return None

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Ошибка сети при получении списка моделей: {req_err}")
        st.error("Ошибка получения списка моделей: Проверьте соединение.")
        return None

    except requests.exceptions.Timeout:
        st.error("Превышено время ожидания ответа от сервера.")
        logger.error("Превышено время ожидания ответа от сервера")
        return None

    except json.JSONDecodeError as json_err:
        logger.error(f"Ошибка декодирования JSON: {json_err}")
        st.error("Ошибка при обработке данных сервера: Неверный формат ответа.")
        return None


def show_models_list(url_server: str) -> None:
    """Функция для отображения на странице списка обученных моделей с возможностью их выбора и удаления."""
    st.subheader("Выбор существующих моделей")
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = ""

    model_info_list = get_models_list(url_server)

    if model_info_list is not None:
        model_names = [model.name for model in model_info_list]
        selected_model_name = st.selectbox(
            "Выберите уже обученную модель",
            model_names,
            index=(
                model_names.index(st.session_state.selected_model_name)
                if st.session_state.selected_model_name in model_names
                else 0
            ),
        )

        selected_model_info = next((model for model in model_info_list if model.name == selected_model_name), None)
        show_model_statistics(selected_model_info)
        if selected_model_info.id != "baseline":
            if st.button(f"Удалить модель {selected_model_name}"):
                delete_model(url_server, selected_model_info.id)
                logger.info(f"Модель {selected_model_name} успешно удалена")
                st.rerun()

        if st.button("Удалить все модели"):
            delete_all_models(url_server)
            logger.info("Все обученные модели успешно удалены")
            st.rerun()


def show_forms_create_model(url_server: str) -> None:
    """Функция для отображения на странице формы подготовки модели для обучения."""
    st.subheader("Создание новой модели SVC и выбор гиперпараметров")

    name_model = st.text_input("Введите название модели")
    param_c = st.slider("Выберите параметр регуляризации:", 0.1, 30.0, 0.1)
    kernel = st.selectbox("Выберите ядро:", ["linear", "poly", "rbf", "sigmoid", "precomputed"])
    probability = st.toggle("Включить оценку вероятности")
    learning_curve = st.toggle("Включить learning_curve")

    fit_request_data = FitRequest(
        config={"svc__C": param_c, "svc__kernel": kernel, "svc__probability": probability},
        with_learning_curve=learning_curve,
        name=name_model,
    )

    fit_json = fit_request_data.model_dump()
    if st.button(":red[**Начать обучение модели**]"):
        with st.spinner("Обучение модели..."):
            logger.info(f"Обучение новой модели {name_model}")

            try:
                response = requests.post(url_server + "models/fit", json=fit_json, timeout=90)
                response_data = json.loads(response.text)
                model_info = ModelInfo(**response_data)
                st.session_state.selected_model_name = model_info.name
                logger.info(f"Модель {name_model} успешно обучена")
                st.rerun()

            except requests.exceptions.HTTPError as http_err:
                st.error("Ошибка сервера при обучении модели.")
                logger.error(f"HTTP ошибка: {http_err}")

            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа от сервера.")
                logger.error("Превышено время ожидания ответа от сервера")

            except requests.exceptions.RequestException as req_err:
                st.error("Ошибка сети при обучении модели.")
                logger.error(f"Ошибка сети: {req_err}")

            except json.JSONDecodeError as json_err:
                st.error("Ошибка при обработке ответа сервера.")
                logger.error(f"Ошибка декодирования JSON: {json_err}")


def model_training_page(url_server: str):
    """Функция для заполнения страницы с подготовкой модели для обучения."""
    show_models_list(url_server)
    change_models_learning_curve()
    show_forms_create_model(url_server)
