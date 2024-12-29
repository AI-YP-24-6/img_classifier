import os
import sys
from dataclasses import dataclass
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ConfigDict

from Backend.app.api.v1.api_route import router_dataset, router_models

# Импорт нужен для работы baseline
from Backend.app.services.pipeline import HogTransformer  # pylint: disable=unused-import # noqa: F401

LOG_FOLDER = "logs"


def configure_logging():
    """
    Конфигурация loguru для правильного записывания логов и работы с uvicorn
    """
    logger.remove()
    os.makedirs(LOG_FOLDER, exist_ok=True)
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|<level>{level}</level>| {message}",
    )
    logger.add(
        os.path.join(LOG_FOLDER, "backend.log"),
        colorize=True,
        format="{time} | {level} | {message}",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
    )

    class InterceptHandler:  # pylint: disable=too-few-public-methods
        def write(self, message):
            """
            Интеграция loguru с uvicorn
            """
            if message.strip():
                logger.info(message.strip())

    sys.stderr = InterceptHandler()


configure_logging()


app = FastAPI(
    title="Классификатор изображений фруктов и овощей",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


@dataclass
class StatusResponse(BaseModel):
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
