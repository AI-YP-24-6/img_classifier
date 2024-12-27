from enum import Enum
from typing import Union, Optional, Annotated, Any
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from http import HTTPStatus
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from loguru import logger

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
    id: Annotated[str, "Id модели. Если требуется baseline-модель, то следует использовать Id='baseline'"]


class ModelListResponse(BaseModel):
    models: Annotated[list[ModelInfo], "Список моделей, доступных пользователю"]


class ModelConfiguration(ModelInfo):
    model_name: Annotated[str, "Название модели"]
    hyperparameters: Annotated[Optional[dict[str, Any]], "Гиперпараметры модели"] = None


class DatasetInfo(BaseModel):
    classes: Annotated[dict[str, int], "Информация о количестве изображений в классах"]
    duplicates: Annotated[dict[str, int], "Информация о дубликатах в классах"]


@router.post("/load_dataset", response_model=DatasetInfo, status_code=HTTPStatus.CREATED)
async def fit(file: Annotated[UploadFile, File(..., description="Арихв с классами изображений")]):
    if file.filename.lower().endswith(".zip") == False:
        logger.exception("Неверный формат файла. Должен загружаться zip-архив!")
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
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/fit", response_model=ModelInfo, status_code=HTTPStatus.CREATED)
async def fit(config: Annotated[Optional[dict[str, Any]], "Гиперпараметры модели (опционально)"] = None):
    try:
        preprocess_dataset((64, 64))
        new_model = create_model(config)
        images, labels = load_colored_images_and_labels()
        new_model.fit(images, labels)
        model_id = str(uuid.uuid4())
        models[model_id] = {'model': new_model, 'type': ModelType.custom, 'hyperparameters': config}
        return ModelInfo(id=model_id, type=ModelType.custom, hyperparameters=config)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/predict", response_model=PredictionResponse, status_code=HTTPStatus.OK)
async def predict(file: Annotated[UploadFile, File(..., description="Файл изображения для предсказания")]):
    global active_model
    if active_model is None:
        logger.exception("Не выбрана модель")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        return PredictionResponse(prediction=active_model.predict([image])[0])
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/load_baseline", response_model=ModelInfo, status_code=HTTPStatus.OK)
async def load():
    global active_model
    if 'baseline' in models:
        active_model = models['baseline']['model']
        return ModelInfo(id="baseline", hyperparameters=models['baseline']['hyperparameters'], type=ModelType.baseline)
    else:
        baseline = load_model()
        model_info = {
            'id': 'baseline',
            'type': 'baseline',
            'hyperparameters': {'pca__n_components': 0.6},
            'model': baseline
        }
        active_model = baseline
        models['baseline'] = model_info
        return ModelInfo(id="baseline", hyperparameters=model_info['hyperparameters'], type=ModelType.baseline)


@router.post("/load", response_model=ModelInfo, status_code=HTTPStatus.OK)
async def load(request: LoadRequest):
    global active_model
    if request.id in models:
        active_model = models[request.id]['model']
        return ModelInfo(id=request.id, hyperparameters=models[request.id]['hyperparameters'], type=models[request.id]['type'])
    else:
        logger.exception(f"Модель '{request.id}' не была найдена!")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Модель '{request.id}' не была найдена!")


@router.post("/unload", response_model=list[ApiResponse], status_code=HTTPStatus.OK)
async def unload():
    global active_model
    active_model = None
    return [ApiResponse(message="true")]


@router.get("/list_models", response_model=dict[str, ModelInfo], status_code=HTTPStatus.OK)
async def list_models():
    return {key: ModelInfo(id=key, type=models[key]['type'], hyperparameters=models[key]['hyperparameters']) for key in models.keys()}


@router.delete("/remove/{model_id}", response_model=dict[str, ModelInfo], status_code=HTTPStatus.OK)
async def remove(model_id: Annotated[str, "Id модели, которую нужно удалить"]):
    if model_id in models.keys():
        del models[model_id]
        return {key: ModelInfo(id=key, type=models[key]['type'], hyperparameters=models[key]['hyperparameters']) for key in models.keys()}
    else:
        logger.exception(f"Нет модели с id '{model_id}'")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Нет модели с id '{model_id}'")


@router.delete("/remove_all", response_model=list[ApiResponse], status_code=HTTPStatus.OK)
async def remove_all():
    global models
    models = {}
    return [ApiResponse(message=f"Все модели удалены")]
