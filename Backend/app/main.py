from dataclasses import dataclass
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from Backend.app.api.models import ModelType
from Backend.app.api.v1.api_route import models, router_dataset, router_models

# Импорт нужен для работы baseline
from Backend.app.services.model_loader import load_model
from Backend.app.services.pipeline import HogTransformer  # pylint: disable=unused-import # noqa: F401
from logs.logger_config import configure_server_logging

configure_server_logging()


app = FastAPI(
    title="Классификатор изображений фруктов и овощей",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


@app.on_event("startup")
def load_baseline_model():
    """
    Загрузка baseline-модели при старте сервера
    """
    baseline_model = load_model()
    models["baseline"] = {
        "id": "baseline",
        "type": ModelType.baseline,
        "hyperparameters": {"pca__n_components": 0.6},
        "model": baseline_model,
        "name": "Baseline",
        "learning_curve": None,
    }


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
    uvicorn.run("main:app", host="127.0.0.1", port=54545, reload=True)
