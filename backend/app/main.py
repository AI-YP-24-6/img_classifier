from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, ConfigDict

from backend.app.api.v1.api_route import router
from backend.app.services.model_loader import load_model
from backend.app.services.pipeline import HogTransformer


app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get("/", response_model=StatusResponse, status_code=HTTPStatus.OK)
async def root():
    return StatusResponse(status="App healthy")


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.1.1", port=8089, reload=True)
