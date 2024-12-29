import os
import sys
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ConfigDict

from Backend.app.api.v1.api_route import router_models, router_dataset

LOG_FOLDER = "logs"


def configure_logging():
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

    # Интеграция loguru с uvicorn
    class InterceptHandler:
        def write(self, message):
            if message.strip():
                logger.info(message.strip())

    sys.stderr = InterceptHandler()


configure_logging()


app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "App healthy"}]})


@app.get("/", response_model=StatusResponse, status_code=HTTPStatus.OK)
async def root():
    return StatusResponse(status="App healthy")


app.include_router(router_dataset)
app.include_router(router_models)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=54545, reload=True)
