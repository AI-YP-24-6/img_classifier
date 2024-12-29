import time

from loguru import logger
from sklearn.pipeline import Pipeline


def train_model(model: Pipeline, images: list, labels: list, model_id: str, manager: dict):
    """
    Обучение модели и ее возврат в Manager
    """
    try:
        start_time = time.time()
        model.fit(images, labels)
        logger.info(f"Модель обучилась за {time.time() - start_time} сек.")
        manager[model_id] = model
    except Exception as e:
        raise ValueError(f"Ошибка обучения модели: {e}") from e
