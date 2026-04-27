from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from rossmann_mlops.config import load_config
from rossmann_mlops.predict import PredictionInputError, Predictor


class PredictionRow(BaseModel):
    Store: int = Field(..., ge=1)
    DayOfWeek: int = Field(..., ge=1, le=7)
    Date: str
    Open: int = Field(..., ge=0, le=1)
    Promo: int = Field(..., ge=0, le=1)
    StateHoliday: str
    SchoolHoliday: int = Field(..., ge=0, le=1)


class PredictionRequest(BaseModel):
    records: list[PredictionRow]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    paths = config.get("paths", {})
    app.state.predictor = Predictor(
        model_path=paths["model_file"],
        store_data_path=paths["store_data"],
        artifacts_dir=paths.get("artifacts_dir"),
    )
    yield


app = FastAPI(title="Rossmann Sales Forecast API", version="0.1.0", lifespan=lifespan)

REQUEST_COUNT = Counter(
    "rossmann_api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "rossmann_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path
    start = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        REQUEST_COUNT.labels(method=method, path=path, status_code=str(status_code)).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(perf_counter() - start)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "message": "Invalid request payload"})


@app.exception_handler(PredictionInputError)
async def prediction_input_exception_handler(_: Request, exc: PredictionInputError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc), "message": "Invalid prediction input"})


@app.exception_handler(ValueError)
async def value_error_exception_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc), "message": "Request could not be processed"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    records = [row.model_dump() for row in payload.records]
    predictions = app.state.predictor.predict(records)
    return {"predictions": predictions, "count": len(predictions)}
