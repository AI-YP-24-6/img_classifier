from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, ConfigDict

from api.v1.api_route import router
from backend.app.services.model_loader import load_model
from backend.app.services.pipeline import HogTransformer

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

baseline = None


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get("/", response_model=StatusResponse, status_code=HTTPStatus.OK)
async def root():
    # Реализуйте метод получения информации о статусе сервиса.
    return StatusResponse(status="App healthy")


# Реализуйте роутер с префиксом /api/v1/models
app.include_router(router)

if __name__ == "__main__":
    baseline = load_model()
    uvicorn.run("main:app", host="0.0.0.0", port=555, reload=True)
