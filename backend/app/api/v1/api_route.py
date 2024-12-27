import json
from enum import Enum
from typing import Union, Optional, Annotated, Any
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from http import HTTPStatus
from pydantic import BaseModel

from sklearn.pipeline import Pipeline
import numpy as np

from backend.app.services.analysis import classes_info, duplicates_info
from backend.app.services.model_loader import load_model
from backend.app.services.pipeline import create_model
from backend.app.services.preprocessing import load_colored_images_and_labels, preprocess_archive, preprocess_dataset, preprocess_image


models: dict[str, Any] = {}
active_model: Union[Pipeline, None] = None


router = APIRouter(prefix="/api/v1/models")


class ApiResponse(BaseModel):
    message: str
    data: Union[dict, None] = None


class PredictionResponse(BaseModel):
    prediction: Annotated[str, "Предсказание класса изображения"]


class ModelType(Enum):
    baseline = 'baseline'
    custom = 'custom'


class ModelInfo(BaseModel):
    id: Annotated[str, "Id модели"]
    hyperparameters: Annotated[Optional[dict[str, Any]], "Гиперпараметры модели"]
    type: Annotated[ModelType, "Тип модели"]


class LoadRequest(BaseModel):
    type: Annotated[ModelType, "Тип модели"]


class ModelListResponse(BaseModel):
    models: Annotated[list[ModelInfo], "Список моделей, доступных пользователю"]


class ModelConfiguration(ModelInfo):
    model_name: Annotated[str, "Название модели"]
    hyperparameters: Annotated[Optional[dict[str, Any]], "Гиперпараметры модели"] = None


class DatasetInfo(BaseModel):
    classes: dict[str, int]
    duplicates: dict[str, int]


@router.post("/load_dataset", response_model=DatasetInfo, status_code=HTTPStatus.CREATED)
async def fit(file: Annotated[UploadFile, File(..., description="Арихв с классами изображений")]):
    if file.filename.lower().endswith(".zip") == False:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Неверный формат файла. Должен загружаться zip-архив!"
        )
    try:
        archive = await file.read()
        preprocess_archive(archive)
        classes = classes_info()
        duplicates = duplicates_info()
        return DatasetInfo(classes=classes, duplicates=duplicates)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/fit", response_model=ModelInfo, status_code=HTTPStatus.CREATED)
async def fit(config: Annotated[Optional[str], Form(..., description="Гиперпараметры модели (опционально)")]):
    try:
        params = json.loads(config)
        if params is not None and params is not dict:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Не удалось распознать гиперпараметры"
            )
        preprocess_dataset()
        new_model = create_model(params)
        images, labels = load_colored_images_and_labels()
        new_model.fit(images, labels)
        model_id = str(uuid.uuid4())
        models[model_id] = {'model': new_model, 'type': ModelType.custom, 'hyperparameters': params}
        return ModelInfo(id=model_id, type=ModelType.custom, hyperparameters=params)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/predict", response_model=PredictionResponse, status_code=HTTPStatus.OK)
async def predict(file: Annotated[UploadFile, File(..., description="Файл изображения для предсказания")]):
    global active_model
    if active_model is None:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        return PredictionResponse(prediction=active_model.predict([image])[0])
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/load", response_model=ModelInfo, status_code=HTTPStatus.OK)
async def load(request: LoadRequest):
    global active_model
    if request.id in models:
        active_model = models[request.id]
        return [ApiResponse(message=f"Модель '{request.id}' загружена!")]
    else:
        return [ApiResponse(message=f"Модель '{request.id}' не была найдена!")]


@router.post("/unload", response_model=list[ApiResponse], status_code=HTTPStatus.OK)
async def unload(request: ModelInfo):
    global active_model
    active_model = None
    return [ApiResponse(message="true")]


@router.get("/list_models", response_model=dict[str, str], status_code=HTTPStatus.OK)
async def list_models():
    return {{"id": key} for key in models.keys()}


@router.delete("/remove/{model_id}", response_model=list[ApiResponse], status_code=HTTPStatus.OK)
async def remove(model_id: str):
    # Удаление обученной модели из списка по id модели
    if model_id in models.keys():
        del models[model_id]
        return [ApiResponse(message=f"true")]
    else:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Нет модели с id '{model_id}'")


@router.delete("/remove_all", response_model=list[ApiResponse], status_code=HTTPStatus.OK)
async def remove_all():
    global models
    models = {}
    return [ApiResponse(message=f"true")]
