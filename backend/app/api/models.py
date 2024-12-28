from pydantic import BaseModel
from typing import Union, Optional, Annotated, Any
from enum import Enum


class ApiResponse(BaseModel):
    message: str
    data: Union[dict, None] = None


class PredictionResponse(BaseModel):
    prediction: Annotated[str, "Предсказание класса изображения"]


class ModelType(Enum):
    baseline = 'baseline'
    custom = 'custom'


class LearningCurvelInfo(BaseModel):
    train_sizes: Annotated[list[int], "Количество примеров обучения"]
    train_scores: Annotated[list[list[float]],
                            "Результаты тренировочных сетов"]
    test_scores: Annotated[list[list[float]], "Результаты тестовых сетов"]


class ModelInfo(BaseModel):
    id: Annotated[str, "Id модели"]
    hyperparameters: Annotated[dict[str, Any], "Гиперпараметры модели"]
    type: Annotated[ModelType, "Тип модели"]
    learning_curve: Annotated[Optional[LearningCurvelInfo],
                              "Данные для кривой обучения модели"]
    name: Annotated[str, "Название модели"]


class LoadRequest(BaseModel):
    id: Annotated[str, "Id модели. Если требуется baseline-модель, то следует использовать Id='baseline'"]


class FitRequest(BaseModel):
    config: Annotated[dict[str, Any], "Гиперпараметры модели (опционально)"]
    with_learning_curve: Annotated[bool,
                                   "Сохранять ли для модели кривую обучения"]
    name: Annotated[str, "Название модели"]


class ModelListResponse(BaseModel):
    models: Annotated[list[ModelInfo],
                      "Список моделей, доступных пользователю"]


class TableModel(BaseModel):
    columns: list[Any]
    rows: list[list[Any]]


class DatasetInfo(BaseModel):
    classes: Annotated[dict[str, int],
                       "Информация о количестве изображений в классах"]
    duplicates: Annotated[dict[str, int], "Информация о дубликатах в классах"]
    sizes: Annotated[TableModel,
                     "Список размеров изображений"]
    colors: Annotated[TableModel, "Список цветов по каналам изображений"]
