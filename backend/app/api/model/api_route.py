import time
from enum import Enum
from typing import Union, Dict, List, Optional, Any

import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from http import HTTPStatus
from pydantic import BaseModel

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from backend.app.services.model_loader import load_model
from backend.app.services.preprocessing import preprocess_image
from backend.app.services.pipeline import HogTransformer, create_model

import numpy as np
models: Dict[str, Union[LogisticRegression, LinearRegression]] = {}
active_model: Union[Pipeline, None] = None

router = APIRouter(prefix="/api/v1/models")

class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None

class PredictionsResponse(BaseModel):
    predictions: List[float]

class ModelListResponse(BaseModel):
    # Список словарей
    models:List[Dict[str, str]]

class ModelType(str, Enum):
    linear = 'linear'
    logistic = 'logistic'

class ModelInfo(BaseModel):
    id: str

class ModelConfiguration(ModelInfo):
    ml_model_type: ModelType
    hyperparameters: Optional[Dict[str, Union[float, int, str, bool]]] = None

class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    config: ModelConfiguration

class PredictRequest(BaseModel):
    X: List[List[float]]
    id: str

# API endpoints
@router.post("/fit", response_model=ApiResponse, status_code=HTTPStatus.CREATED)
async def fit(config: str = Form(...), file: UploadFile = File(...)):

    new_model = create_model()

@router.post("/load", response_model=List[ApiResponse], status_code=HTTPStatus.OK)
async def load(request: ModelInfo):
    global active_model
    if request.id in models:
        active_model = models[request.id]
        return [ApiResponse(message = f"Model '{request.id}' loaded")]
    else:
        return [ApiResponse(message = f"No '{request.id}' in models")]

@router.post("/unload", response_model=List[ApiResponse], status_code=HTTPStatus.OK)
async def unload(request:ModelInfo):
    global active_model
    active_model = None
    return [ApiResponse(message="true")]


@router.post("/predict", response_model=str, status_code=HTTPStatus.OK)
async def predict(file:UploadFile = File(...)):
    global active_model
    active_model = load_model()
    if active_model is None:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        return active_model.predict([image])[0]
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.get("/list_models", response_model=List[Dict[str, str]], status_code=HTTPStatus.OK)
async def list_models():
    # Реализуйте получения списка обученных моделей
    return [{"id": key} for key in models.keys()]

@router.delete("/remove/{model_id}", response_model=List[ApiResponse], status_code=HTTPStatus.OK)
async def remove(model_id:str):
    # Удаление обученной модели из списка по id модели
    if model_id in models.keys():
        del models[model_id]
        return [ApiResponse(message=f"true")]
    else:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"No model with id '{model_id}'")

# Реализуйте Delete метод remove_all
@router.delete("/remove_all", response_model=List[ApiResponse], status_code=HTTPStatus.OK)
async def remove_all():
    global models
    models = {}
    return [ApiResponse(message=f"true")]
