from typing import Union, Annotated, Any
from uuid import uuid4
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from http import HTTPStatus
from fastapi.responses import StreamingResponse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from loguru import logger

from backend.app.api.models import ApiResponse, DatasetInfo, FitRequest, LearningCurvelInfo, LoadRequest, ModelInfo, ModelType, PredictionResponse
from backend.app.services.analysis import classes_info, colors_info, duplicates_info, sizes_info
from backend.app.services.model_loader import load_model
from backend.app.services.pipeline import create_model
from backend.app.services.preview import preview_dataset, remove_preview
from backend.app.services.preprocessing import load_colored_images_and_labels, preprocess_archive, preprocess_dataset, preprocess_image


models: dict[str, Any] = {}
active_model: Union[Pipeline, None] = None
dataset_info: Union[dict[str, Any], None] = None


router = APIRouter(prefix="/api/v1/models")


@router.post("/load_dataset", response_model=DatasetInfo, status_code=HTTPStatus.CREATED)
async def fit(file: Annotated[UploadFile, File(..., description="Арихв с классами изображений")]):
    if file.filename.lower().endswith(".zip") == False:
        logger.exception(
            "Неверный формат файла. Должен загружаться zip-архив!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Неверный формат файла. Должен загружаться zip-архив!"
        )
    global dataset_info
    try:
        archive = await file.read()
        # разархивировал каринки
        preprocess_archive(archive)
        # удалил прошлое превью, если было
        remove_preview()
        classes = classes_info()
        duplicates = duplicates_info()
        sizes = sizes_info()
        colors = colors_info()
        dataset_info = {
            'classes': classes,
            'duplicates': duplicates,
            'sizes': sizes,
            'colors': colors
        }
        return DatasetInfo(
            classes=classes,
            duplicates=duplicates,
            sizes=sizes,
            colors=colors
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.get("/dataset_info", response_model=DatasetInfo, status_code=HTTPStatus.OK)
async def get_dataset_info():
    global dataset_info
    if dataset_info is None:
        # Не логгируется, т.к. не ошибка
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Нет загруженного набора данных!"
        )
    return DatasetInfo(
        classes=dataset_info['classes'],
        duplicates=dataset_info['duplicates'],
        sizes=dataset_info['sizes'],
        colors=dataset_info['colors']
    )


@router.get("/dataset_samples", response_class=StreamingResponse, status_code=HTTPStatus.OK)
async def dataset_samples():
    if dataset_info is None:
        logger.exception("Нет загруженного набора данных!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Нет загруженного набора данных!"
        )
    try:
        buffer = preview_dataset(3)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/fit", response_model=ModelInfo, status_code=HTTPStatus.CREATED)
async def fit(request: Annotated[FitRequest, "Параметры для обучения модели"]):
    try:
        preprocess_dataset((64, 64))
        new_model = create_model(request.config)
        images, labels = load_colored_images_and_labels()

        curve = None
        if request.with_learning_curve:
            train_sizes, train_scores, test_scores = learning_curve(
                new_model, images, labels, cv=5, scoring='f1_macro', train_sizes=[0.3, 0.6, 0.9])
            curve = LearningCurvelInfo(
                test_scores=test_scores, train_scores=train_scores, train_sizes=train_sizes)

        new_model.fit(images, labels)
        model_id = str(uuid4())
        models[model_id] = {'model': new_model, 'type': ModelType.custom, 'name': request.name,
                            'hyperparameters': request.config, 'learning_curve': curve}
        return ModelInfo(
            name=request.name,
            id=model_id,
            type=ModelType.custom,
            hyperparameters=request.config,
            learning_curve=curve
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/predict", response_model=PredictionResponse, status_code=HTTPStatus.OK)
async def predict(file: Annotated[UploadFile, File(..., description="Файл изображения для предсказания")]):
    global active_model
    if active_model is None:
        logger.exception("Не выбрана модель")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                            detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        return PredictionResponse(prediction=active_model.predict([image])[0])
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/load_baseline", response_model=ModelInfo, status_code=HTTPStatus.OK)
async def load_baseline():
    global active_model
    if 'baseline' in models:
        active_model = models['baseline']['model']
        return ModelInfo(
            id='baseline',
            hyperparameters=models['baseline']['hyperparameters'],
            type=ModelType.baseline,
            name='Baseline',
            learning_curve=None
        )
    else:
        baseline = load_model()
        model_info = {
            'id': 'baseline',
            'type': ModelType.baseline,
            'hyperparameters': {'pca__n_components': 0.6},
            'model': baseline,
            'name': 'Baseline',
            'learning_curve': None
        }
        active_model = baseline
        models['baseline'] = model_info
        return ModelInfo(
            id='baseline',
            hyperparameters=model_info['hyperparameters'],
            type=ModelType.baseline,
            name="Baseline",
            learning_curve=None
        )


@router.post("/load", response_model=ModelInfo, status_code=HTTPStatus.OK)
async def load(request: LoadRequest):
    global active_model
    if request.id in models:
        model = models[request.id]
        active_model = model['model']
        return ModelInfo(
            id=request.id,
            hyperparameters=model['hyperparameters'],
            type=model['type'],
            learning_curve=model['learning_curve'],
            name=model['name']
        )
    else:
        logger.exception(f"Модель '{request.id}' не была найдена!")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Модель '{
                            request.id}' не была найдена!")


@router.post("/unload", response_model=ApiResponse, status_code=HTTPStatus.OK)
async def unload():
    global active_model
    active_model = None
    return ApiResponse(message="Модель выгружена из памяти")


@router.get("/list_models", response_model=dict[str, ModelInfo], status_code=HTTPStatus.OK)
async def list_models():
    return {key: ModelInfo(
        id=key,
        type=models[key]['type'],
        hyperparameters=models[key]['hyperparameters'],
        learning_curve=models[key]['learning_curve'],
        name=models[key]['name']
    ) for key in models.keys()}


@router.get("/info/{model_id}", response_model=ModelInfo, status_code=HTTPStatus.OK)
async def model_info(model_id: Annotated[str, "Id модели"]):
    if model_id in models:
        model = models[model_id]
        return ModelInfo(
            id=model['id'],
            type=model['type'],
            hyperparameters=model['hyperparameters'],
            learning_curve=model['learning_curve'],
            name=model['name']
        )
    else:
        logger.exception(f"Модель '{model_id}' не была найдена!")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Модель '{
                            model_id}' не была найдена!")


@router.delete("/remove/{model_id}", response_model=dict[str, ModelInfo], status_code=HTTPStatus.OK)
async def remove(model_id: Annotated[str, "Id модели, которую нужно удалить"]):
    if model_id in models.keys():
        del models[model_id]
        return {key: ModelInfo(
            id=key,
            type=models[key]['type'],
            hyperparameters=models[key]['hyperparameters'],
            learning_curve=models[key]['learning_curve'],
            name=models[key]['name']
        ) for key in models.keys()}
    else:
        logger.exception(f"Нет модели с id '{model_id}'")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                            detail=f"Нет модели с id '{model_id}'")


@router.delete("/remove_all", response_model=ApiResponse, status_code=HTTPStatus.OK)
async def remove_all():
    global models
    models = {}
    return ApiResponse(message=f"Все модели удалены")
