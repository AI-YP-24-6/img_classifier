from http import HTTPStatus
from multiprocessing import Manager, Process
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from Backend.app.api.models import (
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
from Backend.app.services.analysis import check_dataset_uploaded, classes_info, colors_info, duplicates_info, sizes_info
from Backend.app.services.model_loader import load_model
from Backend.app.services.model_trainer import train_model
from Backend.app.services.pipeline import create_model
from Backend.app.services.preprocessing import (
    load_colored_images_and_labels,
    preprocess_archive,
    preprocess_dataset,
    preprocess_image,
)
from Backend.app.services.preview import preview_dataset, remove_preview

models: dict[str, Any] = {}
active_model: dict[str, Pipeline | None] = {"model": None, "info": None}
dataset_info = DatasetInfo(
    is_empty=True,
    classes={},
    duplicates={},
    sizes=TableModel(columns=[], rows=[]),
    colors=TableModel(columns=[], rows=[]),
)


router_models = APIRouter(prefix="/api/v1/models")
router_dataset = APIRouter(prefix="/api/v1/dataset")


@router_dataset.post(
    "/load",
    response_model=Annotated[DatasetInfo, "Информация о датасете"],
    status_code=HTTPStatus.CREATED,
    description="Загрузка датасета",
)
async def load_dataset(file: Annotated[UploadFile, File(..., description="Архив с классами изображений")]):
    """
    Загрузка датасета.
    На вход должен подаваться архив, содержащий папки с изображениями классов.
    """
    if file.filename.lower().endswith(".zip") is False:
        logger.exception("Неверный формат файла. Должен загружаться zip-архив!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Неверный формат файла. Должен загружаться zip-архив!"
        )
    try:
        archive = await file.read()
        # разархивировал картинки
        preprocess_archive(archive)
        # удалил прошлое превью, если было
        remove_preview()
        dataset_info.classes = classes_info()
        dataset_info.duplicates = duplicates_info()
        dataset_info.sizes = TableModel(rows=sizes_info(), columns=["class", "name", "width", "height"])
        dataset_info.colors = TableModel(
            rows=colors_info(), columns=["class", "name", "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"]
        )
        dataset_info.is_empty = False
        return dataset_info
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e


@router_dataset.get(
    "/info",
    response_model=Annotated[DatasetInfo, "Информация о датасете"],
    status_code=HTTPStatus.OK,
    description="Получение информации о датасете",
)
async def get_dataset_info():
    """
    Получение информации о датасете.
    Возвращается количество изображений в каждом классе, дубли, таблица размеров и цветов
    """
    dataset_uploaded = check_dataset_uploaded()
    if dataset_uploaded is False:
        # Не логгируется, т.к. не ошибка
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Нет загруженного набора данных!")
    if dataset_info.is_empty:
        dataset_info.classes = classes_info()
        dataset_info.duplicates = duplicates_info()
        dataset_info.sizes = TableModel(rows=sizes_info(), columns=["class", "name", "width", "height"])
        dataset_info.colors = TableModel(
            rows=colors_info(), columns=["class", "name", "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"]
        )
        dataset_info.is_empty = False
    return dataset_info


@router_dataset.get(
    "/samples",
    response_class=Annotated[StreamingResponse, "Пример с изображениями"],
    status_code=HTTPStatus.OK,
    description="Изображения из классов",
)
async def dataset_samples():
    """
    Возвращает картинку с примерами изображений по каждому классу
    """
    if dataset_info.is_empty:
        logger.exception("Нет загруженного набора данных!")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Нет загруженного набора данных!")
    try:
        buffer = preview_dataset(3)
        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e


@router_models.post(
    "/fit",
    response_model=Annotated[ModelInfo, "Информация об обученной модели"],
    status_code=HTTPStatus.CREATED,
    description="Обучение модели",
)
async def fit(request: Annotated[FitRequest, "Параметры для обучения модели"]):
    """
    Обучение модели. По истечении 10 секунд обучение прерывается.
    Есть возможность дополнительно получить кривую обучения, указав `with_learning_curve=True`
    Также для обучения модели передаются гиперпараметры вида `pca__` и `svc__`
    """
    manager = Manager()
    model_manager = manager.dict()
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

        model_id = str(uuid4())
        process = Process(target=train_model, args=(new_model, images, labels, model_id, model_manager))
        process.start()
        # Через 10 сек. обучение прервется, т.к. считается долгим
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            raise HTTPException(status_code=HTTPStatus.REQUEST_TIMEOUT, detail="Время обучения модели истекло")

        new_model = model_manager.get(model_id, None)
        if new_model is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Ошибка обучения модели!")

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
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e


@router_models.post(
    "/predict",
    response_model=Annotated[PredictionResponse, "Предсказание"],
    status_code=HTTPStatus.OK,
    description="Предсказание класса",
)
async def predict(file: Annotated[UploadFile, File(..., description="Файл изображения для предсказания")]):
    """
    Предсказание изображенного фрукта или овоща
    """
    if active_model.get("model", None) is None:
        logger.exception("Не выбрана модель")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        model: Pipeline = active_model["model"]
        return PredictionResponse(prediction=model.predict([image])[0])
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e


@router_models.post(
    "/predict_proba",
    response_model=Annotated[ProbabilityResponse, "Предсказание с вероятностью"],
    status_code=HTTPStatus.OK,
    description="Предсказание класса с вероятностью",
)
async def predict_proba(file: Annotated[UploadFile, File(..., description="Файл изображения для предсказания")]):
    """
    Предсказание с вероятностью изображенного фрукта или овоща
    """
    if active_model.get("model", None) is None:
        logger.exception("Не выбрана модель")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Не выбрана модель")
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        model: Pipeline = active_model["model"]
        probability = max(model.predict_proba([image])[0])
        prediction = model.predict([image])[0]
        return ProbabilityResponse(prediction=prediction, probability=probability)
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e


@router_models.post(
    "/load_baseline",
    response_model=Annotated[ModelInfo, "Информация о baseline-модели"],
    status_code=HTTPStatus.OK,
    description="Загрузка baseline-модели",
)
async def load_baseline():
    """
    Загрузка baseline-модели для работы с ней
    В первый раз загружается из picke-файла, затем - из памяти
    """
    if "baseline" in models:
        info = ModelInfo(
            id="baseline",
            hyperparameters=models["baseline"]["hyperparameters"],
            type=ModelType.baseline,
            name="Baseline",
            learning_curve=None,
        )
        active_model["model"] = models["baseline"]["model"]
        active_model["info"] = info
        return info
    baseline = load_model()
    baseline_info = {
        "id": "baseline",
        "type": ModelType.baseline,
        "hyperparameters": {"pca__n_components": 0.6},
        "model": baseline,
        "name": "Baseline",
        "learning_curve": None,
    }
    models["baseline"] = baseline_info
    info = ModelInfo(
        id="baseline",
        hyperparameters=baseline_info["hyperparameters"],
        type=ModelType.baseline,
        name="Baseline",
        learning_curve=None,
    )
    active_model["model"] = baseline
    active_model["info"] = info
    return info


@router_models.post(
    "/load",
    response_model=Annotated[ModelInfo, "Информация о выбранной модели"],
    status_code=HTTPStatus.OK,
    description="Загрузка одной из моделей",
)
async def load(request: LoadRequest):
    """
    Загрузка пользовательской модели для использования. Модель загружается по id
    """
    if request.id not in models:
        logger.exception(f"Модель '{request.id}' не была найдена!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Модель '{request.id}' не была найдена!",
        )
    model = models[request.id]
    info = ModelInfo(
        id=request.id,
        hyperparameters=model["hyperparameters"],
        type=model["type"],
        learning_curve=model["learning_curve"],
        name=model["name"],
    )
    active_model["model"] = model["model"]
    active_model["info"] = info
    return info


@router_models.post(
    "/unload",
    response_model=Annotated[ApiResponse, "Сообщение о выгрузке модели"],
    status_code=HTTPStatus.OK,
    description="Выгрузка модели из памяти",
)
async def unload():
    """
    Выгрузка модели.
    Если модель была выгружена, то предсказания не будут работать пока не загрузят новую модель
    """
    active_model["model"] = None
    active_model["info"] = None
    return ApiResponse(message="Модель выгружена из памяти")


@router_models.get(
    "/list_models",
    response_model=Annotated[dict[str, ModelInfo], "Информация о моделях на сервере"],
    status_code=HTTPStatus.OK,
    description="Получение списка моделей",
)
async def list_models():
    """
    Возврат списка всех доступных моделей
    """
    return {
        model_id: ModelInfo(
            id=model_id,
            type=model["type"],
            hyperparameters=model["hyperparameters"],
            learning_curve=model["learning_curve"],
            name=model["name"],
        )
        for model_id, model in models.items()
    }


@router_models.get(
    "/info/{model_id}",
    response_model=Annotated[ModelInfo, "Информация о модели"],
    status_code=HTTPStatus.OK,
    description="Получение информации о модели",
)
async def model_info(model_id: Annotated[str, "Id модели"]):
    """
    Возвращает информацию по модели с указанным id.
    В информацию входит:
    - id
    - тип модели (baseline/custom)
    - гиперпараметры (какие были использованы при обучении)
    - кривая обучения (если получалась при обучении)
    - пользовательское название модели
    """
    if model_id not in models:
        logger.exception(f"Модель '{model_id}' не была найдена!")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Модель '{model_id}' не была найдена!",
        )
    model = models[model_id]
    return ModelInfo(
        id=model["id"],
        type=model["type"],
        hyperparameters=model["hyperparameters"],
        learning_curve=model["learning_curve"],
        name=model["name"],
    )


@router_models.delete(
    "/remove/{model_id}",
    response_model=Annotated[dict[str, ModelInfo], "Оставшиеся модели"],
    status_code=HTTPStatus.OK,
    description="Удаление модели",
)
async def remove(model_id: Annotated[str, "Id модели, которую нужно удалить"]):
    """
    Удалит модель из памяти, ее больше нельзя будет загрузить для работы.
    """
    if model_id not in models:
        logger.exception(f"Нет модели с id '{model_id}'")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Нет модели с id '{model_id}'")
    del models[model_id]
    return {
        model_id: ModelInfo(
            id=model_id,
            type=model["type"],
            hyperparameters=model["hyperparameters"],
            learning_curve=model["learning_curve"],
            name=model["name"],
        )
        for model_id, model in models.items()
    }


@router_models.delete(
    "/remove_all",
    response_model=Annotated[ApiResponse, "Сообщение об успешном удалении"],
    status_code=HTTPStatus.OK,
    description="Удаление всех моделей",
)
async def remove_all():
    """
    Полностью удалит все модели, очистка списка моделей
    """
    models.clear()
    return ApiResponse(message="Все модели удалены")
