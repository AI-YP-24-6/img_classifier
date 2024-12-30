import os
from contextlib import asynccontextmanager
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from Backend.app.api.models import ModelType
from Backend.app.api.v1.api_route import models, router_dataset, router_models

# Импорт нужен для работы baseline
from Backend.app.services.model_loader import load_model
from Tools.logger_config import configure_server_logging

configure_server_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Логика, выполняющаяся при старте и остановке приложения
    Загрузка baseline-модели при старте сервера
    """
    baseline_model = load_model()
    models["baseline"] = {
        "id": "baseline",
        "type": ModelType.baseline,
        "hyperparameters": {"pca__n_components": 0.6, "svc_probability": True},
        "model": baseline_model,
        "name": "Baseline",
        "learning_curve": None,
    }
    yield


app = FastAPI(
    title="Классификатор изображений фруктов и овощей",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


class StatusResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Статус работы сервиса
    """

    status: str

    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "App healthy"}]})


@app.get("/", response_model=StatusResponse, status_code=HTTPStatus.OK)
async def root():
    """
    Возврат статуса работы сервиса
    """
    return StatusResponse(status="App healthy")


app.include_router(router_dataset)
app.include_router(router_models)

if __name__ == "__main__":
    host = os.getenv("UVICORN_HOST", "127.0.0.1")
    port = int(os.getenv("UVICORN_PORT", "54545"))
    reload = bool(os.getenv("UVICORN_RELOAD", "False"))
    uvicorn.run("main:app", host=host, port=port, reload=reload)
