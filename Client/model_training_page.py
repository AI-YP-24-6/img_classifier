import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from loguru import logger

from Backend.app.api.models import FitRequest, ModelInfo


def plt_learning_curve(train_sizes, train_scores, test_scores):
    """Функция для отображение графика кривых обучения модели."""
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
    plt.ylabel("f1-macro")
    plt.xticks(train_sizes)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    plt.close()
    logger.info("Построен график кривых обучения модели")


def show_model_statistics(model_info):
    """Функция для отображение информации об обученной модели."""
    st.subheader("Информация о модели")
    hyperparams_str = "".join([f"\n- **{key} =** {value}" for key, value in model_info.hyperparameters.items()])
    st.markdown(
        f"""
    **Название модели:** {model_info.name}
    **Гиперпараметры:** {hyperparams_str}
    """,
    )

    logger.info("Для клиента отображена основная информация об обученной модели")

    learning_curve = model_info.learning_curve
    if learning_curve is not None:
        st.subheader("Полученные кривые обучения")
        plt_learning_curve(learning_curve.train_sizes, learning_curve.train_scores, learning_curve.test_scores)


def delete_model(url_server, model_id):
    """Функция для удаления обученной модели."""
    response = requests.delete(url_server + f"models/remove/{model_id}")
    if response.status_code == 200:
        return True
    return False


def delete_all_models(url_server):
    """Функция для удаления всех обученных моделей."""
    response = requests.delete(url_server + "models/remove_all")
    if response.status_code == 200:
        return True
    return False


def get_models_list(url_server):
    """Функция для получения списка всех обученных моделей."""
    try:
        with st.spinner("Загрузка списка моделей..."):
            logger.info("Загрузка списка моделей с сервера")
            model_info_list = []
            response = requests.get(url_server + "models/list_models")
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

    except json.JSONDecodeError as json_err:
        logger.error(f"Ошибка декодирования JSON: {json_err}")
        st.error("Ошибка при обработке данных сервера: Неверный формат ответа.")
        return None


def show_models_list(url_server):
    """Функция для отображения на странице списка обученных моделей с возможностью их выбора и удаления."""
    st.subheader("Выбор существующих моделей")
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = ""

    model_info_list = get_models_list(url_server)

    if len(model_info_list) != 0:
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
                model_info_list = [model for model in model_info_list if model.name != selected_model_name]
                logger.info(f"Модель {selected_model_name} успешно удалена")
                st.rerun()

        if st.button("Удалить все модели"):
            delete_all_models(url_server)
            logger.info("Все обученные модели успешно удалены")
            st.rerun()


def show_forms_create_model(url_server):
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
                st.rerun()
                logger.info(f"Модель {name_model} успешно обучена")

            except requests.exceptions.HTTPError as http_err:
                st.error("Ошибка сервера при обучении модели.")
                logger.error(f"HTTP ошибка: {http_err}")

            except requests.exceptions.RequestException as req_err:
                st.error("Ошибка сети при обучении модели.")
                logger.error(f"Ошибка сети: {req_err}")

            except json.JSONDecodeError as json_err:
                st.error("Ошибка при обработке ответа сервера.")
                logger.error(f"Ошибка декодирования JSON: {json_err}")
            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа сервера.")
                logger.error("Превышено время ожидания ответа сервера")


def model_training_page(url_server):
    """Функция для заполнения страницы с подготовкой модели для обучения."""
    show_models_list(url_server)
    show_forms_create_model(url_server)
