"""Cycle 4 -- FastAPI inference service (MAD 3.3). Wraps the trained
XGBoost model + preprocessor from Cycles 2-3, applying the tuned operating
threshold from Pillar 4.3 at serving time instead of the default 0.5 cutoff.
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.inference import ModelService
from src.api.logging_config import log_requests
from src.api.schemas import BatchRequest, BatchResponse, FlowFeatures, PredictionResponse

app = FastAPI(title="AI-Driven Network Threat & Intrusion Detection System")
app.middleware("http")(log_requests)

model_service = ModelService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict:
    return model_service.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: FlowFeatures) -> dict:
    return model_service.predict_one(flow)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest) -> dict:
    return {"predictions": model_service.predict_batch(batch.flows)}
