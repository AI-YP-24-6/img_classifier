from http import HTTPStatus
from typing import Annotated, Any, Union
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from backend.app.api.models import (
    ApiResponse,
    DatasetInfo,
    FitRequest,
    LearningCurvelInfo,
    LoadRequest,
    ModelInfo,
    ModelType,
    PredictionResponse,
    ProbabilityResponse,
    TableModel,
)
from backend.app.services.analysis import classes_info, colors_info, duplicates_info, sizes_info
from backend.app.services.model_loader import load_model
from backend.app.services.pipeline import create_model
from backend.app.services.preprocessing import (
    load_colored_images_and_labels,
    preprocess_archive,
    preprocess_dataset,
    preprocess_image,
)
from backend.app.services.preview import preview_dataset, remove_preview

models: dict[str, Any] = {}
active_model: Union[Pipeline, None] = None
active_model_info: Union[ModelInfo, None] = None
dataset_info: Union[DatasetInfo, None] = None


router = APIRouter(prefix="/api/v1/models")


@router.post(
    "/load_dataset",
    response_model=Annotated[DatasetInfo, "Информация о датасете"],
    status_code=HTTPStatus.CREATED,
    description="Загрузка датасета",
)
async def fit(file: Annotated[UploadFile, File(..., description="Арихв с классами изображений")]):
    if file.filename.lower().endswith(".zip") == False:
        logger.exception("Неверный формат файла. Должен загружаться zip-архив!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Неверный формат файла. Должен загружаться zip-архив!"
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
        sizes: TableModel = {"rows": sizes_info(), "columns": ["class", "name", "width", "height"]}
        colors: TableModel = {
            "rows": colors_info(),
            "columns": ["class", "name", "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"],
        }
        dataset_info = DatasetInfo(classes=classes, duplicates=duplicates, sizes=sizes, colors=colors)
        return dataset_info
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.get(
    "/dataset_info",
    response_model=Annotated[DatasetInfo, "Информация о датасете"],
    status_code=HTTPStatus.OK,
    description="Получение информации о датасете",
)
async def get_dataset_info():
    global dataset_info
    if dataset_info is None:
        # Не логгируется, т.к. не ошибка
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Нет загруженного набора данных!")
    return dataset_info


@router.get(
    "/dataset_samples",
    response_class=Annotated[StreamingResponse, "Пример с изображениями"],
    status_code=HTTPStatus.OK,
    description="Изображения из классов",
)
async def dataset_samples():
    if dataset_info is None:
        logger.exception("Нет загруженного набора данных!")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Нет загруженного набора данных!")
    try:
        buffer = preview_dataset(3)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post(
    "/fit",
    response_model=Annotated[ModelInfo, "Информация об обученной модели"],
    status_code=HTTPStatus.CREATED,
    description="Обучение модели",
)
async def fit(request: Annotated[FitRequest, "Параметры для обучения модели"]):
    try:
        preprocess_dataset((64, 64))
        new_model = create_model(request.config)
        images, labels = load_colored_images_and_labels()

        curve = None
        if request.with_learning_curve:
            train_sizes, train_scores, test_scores = learning_curve(
                new_model, images, labels, cv=5, scoring="f1_macro", train_sizes=[0.3, 0.6, 0.9]
            )
            curve = LearningCurvelInfo(test_scores=test_scores, train_scores=train_scores, train_sizes=train_sizes)

        new_model.fit(images, labels)
        model_id = str(uuid4())
        models[model_id] = {
            "id": model_id,
            "model": new_model,
            "type": ModelType.custom,
            "name": request.name,
            "hyperparameters": request.config,
            "learning_curve": curve,
        }
        return ModelInfo(
            name=request.name, id=model_id, type=ModelType.custom, hyperparameters=request.config, learning_curve=curve
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post(
    "/predict",
    response_model=Annotated[PredictionResponse, "Предсказание"],
    status_code=HTTPStatus.OK,
    description="Предсказание класса",
)
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


@router.post(
    "/predict_proba",
    response_model=Annotated[ProbabilityResponse, "Предсказание с вероятностью"],
    status_code=HTTPStatus.OK,
    description="Предсказание класса с вероятностью",
)
async def predict(file: Annotated[UploadFile, File(..., description="Файл изображения для предсказания")]):
    global active_model
    if active_model is None:
        logger.exception("Не выбрана модель")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        probability = max(active_model.predict_proba([image])[0])
        prediction = active_model.predict([image])[0]
        return ProbabilityResponse(prediction=prediction, probability=probability)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post(
    "/load_baseline",
    response_model=Annotated[ModelInfo, "Информация о baseline-модели"],
    status_code=HTTPStatus.OK,
    description="Загрузка baiseline-модели",
)
async def load_baseline():
    global active_model
    global active_model_info
    if "baseline" in models:
        active_model = models["baseline"]["model"]
        active_model_info = ModelInfo(
            id="baseline",
            hyperparameters=models["baseline"]["hyperparameters"],
            type=ModelType.baseline,
            name="Baseline",
            learning_curve=None,
        )
    else:
        baseline = load_model()
        model_info = {
            "id": "baseline",
            "type": ModelType.baseline,
            "hyperparameters": {"pca__n_components": 0.6},
            "model": baseline,
            "name": "Baseline",
            "learning_curve": None,
        }
        active_model = baseline
        models["baseline"] = model_info
        active_model_info = ModelInfo(
            id="baseline",
            hyperparameters=model_info["hyperparameters"],
            type=ModelType.baseline,
            name="Baseline",
            learning_curve=None,
        )
    return active_model_info


@router.post(
    "/load",
    response_model=Annotated[ModelInfo, "Информация о выбранной модели"],
    status_code=HTTPStatus.OK,
    description="Загрузка одной из моделей",
)
async def load(request: LoadRequest):
    global active_model
    global active_model_info
    if request.id in models:
        model = models[request.id]
        active_model = model["model"]
        active_model_info = {
            "id": request.id,
            "hyperparameters": model["hyperparameters"],
            "type": model["type"],
            "learning_curve": model["learning_curve"],
            "name": model["name"],
        }
        return ModelInfo(
            id=request.id,
            hyperparameters=model["hyperparameters"],
            type=model["type"],
            learning_curve=model["learning_curve"],
            name=model["name"],
        )
    else:
        logger.exception(f"Модель '{request.id}' не была найдена!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Модель '{
                            request.id}' не была найдена!",
        )


@router.post(
    "/unload",
    response_model=Annotated[ApiResponse, "Сообщение о выгрузке модели"],
    status_code=HTTPStatus.OK,
    description="Выгрузка модели из памяти",
)
async def unload():
    global active_model
    global active_model_info
    active_model = None
    active_model_info = None
    return ApiResponse(message="Модель выгружена из памяти")


@router.get(
    "/list_models",
    response_model=Annotated[dict[str, ModelInfo], "Информация о моделях на сервере"],
    status_code=HTTPStatus.OK,
    description="Получение списка моделей",
)
async def list_models():
    return {
        key: ModelInfo(
            id=key,
            type=models[key]["type"],
            hyperparameters=models[key]["hyperparameters"],
            learning_curve=models[key]["learning_curve"],
            name=models[key]["name"],
        )
        for key in models.keys()
    }


@router.get(
    "/info/{model_id}",
    response_model=Annotated[ModelInfo, "Информация о модели"],
    status_code=HTTPStatus.OK,
    description="Получение информации о модели",
)
async def model_info(model_id: Annotated[str, "Id модели"]):
    if model_id in models:
        model = models[model_id]
        return ModelInfo(
            id=model["id"],
            type=model["type"],
            hyperparameters=model["hyperparameters"],
            learning_curve=model["learning_curve"],
            name=model["name"],
        )
    else:
        logger.exception(f"Модель '{model_id}' не была найдена!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Модель '{
                            model_id}' не была найдена!",
        )


@router.delete(
    "/remove/{model_id}",
    response_model=Annotated[dict[str, ModelInfo], "Оставшиеся модели"],
    status_code=HTTPStatus.OK,
    description="Удалиние модели",
)
async def remove(model_id: Annotated[str, "Id модели, которую нужно удалить"]):
    if model_id in models.keys():
        del models[model_id]
        return {
            key: ModelInfo(
                id=key,
                type=models[key]["type"],
                hyperparameters=models[key]["hyperparameters"],
                learning_curve=models[key]["learning_curve"],
                name=models[key]["name"],
            )
            for key in models.keys()
        }
    else:
        logger.exception(f"Нет модели с id '{model_id}'")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Нет модели с id '{model_id}'")


@router.delete(
    "/remove_all",
    response_model=Annotated[ApiResponse, "Сообщение об успешном удалении"],
    status_code=HTTPStatus.OK,
    description="Удаление всех моделей",
)
async def remove_all():
    global models
    models = {}
    return ApiResponse(message=f"Все модели удалены")
