import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.svm import SVC
import numpy as np


# def plt_learning_curve():
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

#     common_params = {
#         "X": X,
#         "y": y,
#         "train_sizes": np.linspace(0.1, 1.0, 5),
#         "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
#     "score_type": "both",
#     "n_jobs": 4,
#         "line_kw": {"marker": "o"},
#         "std_display_style": "fill_between",
#         "score_name": "Accuracy",
#     }
    
#     model = SVC()
#     for ax_idx, estimator in enumerate(model):
#         LearningCurveDisplay.from_estimator(model, **common_params, ax=ax[ax_idx])
#         handles, label = ax[ax_idx].get_legend_handles_labels()
#         ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
#         ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")
    
def model_training_page():
    st.header("Создание новой модели SVC и выбор гиперпараметров")
    
    name_model = st.text_input("Введите название модели")
    param_c = st.slider("Выберите параметр регуляризации:", 0.1, 30.0, 0.1)
    kernel = st.selectbox("Выберите ядро:", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    probability = st.checkbox("Включить оценку вероятности")
    
    # request fit model
    
    st.success("Модель успешно создана!")
    
    st.subheader("Информация о модели")
    
    st.markdown(f"""
        - **Название модели:** {name_model}
        - **Параметр C =** {param_c}
        - **Ядро:** {kernel}
        - **Оценка вероятности:** {probability}.
            """)
    
    st.subheader("Поученные кривые обучения")
    # plt_learning_curve()
    
    