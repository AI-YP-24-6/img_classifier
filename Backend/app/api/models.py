from enum import Enum
from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel


class ApiResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Ответ сервера на запрос с клиента
    """

    message: str
    data: Union[dict, None] = None


class PredictionResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Предсказание класса изображения
    """

    prediction: Annotated[str, "Предсказание класса изображения"]


class ProbabilityResponse(PredictionResponse):  # pylint: disable=too-few-public-methods
    """
    Предсказание класса изображения с вероятностью
    """

    probability: Annotated[float, "Вероятность предсказанного класса"]


class ModelType(Enum):  # pylint: disable=too-few-public-methods
    """
    Типы моделей
    """

    baseline = "baseline"
    custom = "custom"


class LearningCurvelInfo(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Информация о кривой обучения модели
    """

    train_sizes: Annotated[list[int], "Количество примеров обучения"]
    train_scores: Annotated[list[list[float]], "Результаты тренировочных сетов"]
    test_scores: Annotated[list[list[float]], "Результаты тестовых сетов"]


class ModelInfo(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Информация об обученной модели
    """

    id: Annotated[str, "Id модели"]
    hyperparameters: Annotated[dict[str, Any], "Гиперпараметры модели"]
    type: Annotated[ModelType, "Тип модели"]
    learning_curve: Annotated[Optional[LearningCurvelInfo], "Данные для кривой обучения модели"]
    name: Annotated[str, "Название модели"]


class LoadRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Запрос на загрузку обученной модели
    """

    id: Annotated[str, "Id модели. Если требуется baseline-модель, то следует использовать Id='baseline'"]


class FitRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Запрос на обучение модели с указанием названия модели, гиперпараметров и сохранения кривой обучения
    """

    config: Annotated[dict[str, Any], "Гиперпараметры модели (опционально)"]
    with_learning_curve: Annotated[bool, "Сохранять ли для модели кривую обучения"]
    name: Annotated[str, "Название модели"]


class TableModel(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Модель таблицы, содержащая ее столбцы и строки
    """

    columns: Annotated[list[Any], "Колонки таблицы"]
    rows: Annotated[list[list[Any]], "Строки таблицы"]


class DatasetInfo(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Информация о датасете:
    - количестве изображений в каждом классе
    - дубликаты в каждом классе
    - таблица размеров
    - таблица цветов
    - загружен датасет или нет
    """

    classes: Annotated[dict[str, int], "Информация о количестве изображений в классах"]
    duplicates: Annotated[dict[str, int], "Информация о дубликатах в классах"]
    sizes: Annotated[TableModel, "Таблица размеров изображений"]
    colors: Annotated[TableModel, "Таблица цветов по каналам изображений"]
    is_empty: Annotated[bool, "Флаг, что датасет не загружен"]