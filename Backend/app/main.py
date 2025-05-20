from contextlib import asynccontextmanager
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from Backend.app.api.models import ModelType
from Backend.app.api.v1.api_route import models, router_dataset, router_models
from Backend.app.services.model_loader import load_model  # Импорт нужен для работы baseline
from Tools.logger_config import configure_server_logging


class Settings(BaseSettings):
    """
    Класс загрузки и валидации настроек из .env
    """

    model_config = SettingsConfigDict(
        env_file="../../.env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )
    uvicorn_host: str = Field(
        default="127.0.0.1", validate_default=False, pattern="[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}"
    )
    uvicorn_port: int = Field(default=54545, validate_default=False, le=65535, ge=0)
    uvicorn_reload: bool = False


settings = Settings()

configure_server_logging("../../logs/")


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


class StatusResponse(BaseModel):
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
    uvicorn.run(
        "main:app",
        host=settings.uvicorn_host,
        port=settings.uvicorn_port,
        reload=settings.uvicorn_reload,
        log_config=None,
    )
