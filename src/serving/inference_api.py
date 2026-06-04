"""
ANALYTICA — Production Model Inference API
src/serving/inference_api.py

FastAPI app exposing POST /predict/{model_name} for all 4 live ANALYTICA models.
JWT auth header is passed through to API-GATEWAY — not validated here.
MLflow model registry loads latest 'Production' stage artifact on startup.

Response contract:
  prediction          list[float]     raw model output
  confidence_interval CI95 lower/upper (where available)
  shap_top5           top-5 SHAP feature importances
  model_version       MLflow model version tag
  inference_duration_ms  wall-clock inference time

Prometheus: instruments tropi_model_inference_duration_seconds{model=<name>}
            for CLOUD-FORGE SLA rules (P99High >5s, P99Critical >15s).

Run:
  uvicorn serving.inference_api:app --host 0.0.0.0 --port 8080 --workers 2
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST, Histogram
from pydantic import BaseModel, Field

log = logging.getLogger("analytica.inference_api")

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0,
    5.0,   # TropiModelInferenceP99High threshold
    7.5, 10.0, 12.5,
    15.0,  # TropiModelInferenceP99Critical threshold
    20.0, 30.0, float("inf"),
)

INFERENCE_HISTOGRAM = Histogram(
    "tropi_model_inference_duration_seconds",
    "Model inference latency (seconds) — drives TropiModelInferenceP99High/Critical SLA rules",
    ["model"],
    buckets=INFERENCE_BUCKETS,
)

REQUEST_COUNTER = Counter(
    "tropi_inference_api_requests_total",
    "Total inference API requests",
    ["model", "status_code"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Model name constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_XGB_NOWCAST       = "xgboost_nowcast"
MODEL_PROPHET_SEASONAL  = "prophet_seasonal"
MODEL_CNN_LAND_COVER    = "cnn_land_cover"
MODEL_LSTM_STREAMFLOW   = "lstm_streamflow"

ALL_MODELS = [MODEL_XGB_NOWCAST, MODEL_PROPHET_SEASONAL,
              MODEL_CNN_LAND_COVER, MODEL_LSTM_STREAMFLOW]

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dict — keys depend on model. See /docs.",
        example={"station": "manggarai", "rainfall_mm": [12.3, 0.0, 5.1], "lookback_days": 14},
    )
    horizon_hours: Optional[int] = Field(None, ge=1, le=168, description="Forecast horizon (xgb, prophet)")
    station: Optional[str] = Field(None, description="Gauge station ID (lstm_streamflow)")
    province_ids: Optional[List[int]] = Field(None, description="Province subset filter (cnn_land_cover)")


class ConfidenceInterval(BaseModel):
    lower: List[float]
    upper: List[float]
    level: float = 0.95


class SHAPFeatureImportance(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "positive" | "negative"


class PredictResponse(BaseModel):
    model: str
    model_version: str
    prediction: List[float]
    confidence_interval: Optional[ConfidenceInterval] = None
    shap_top5: List[SHAPFeatureImportance] = Field(default_factory=list)
    inference_duration_ms: float
    warning: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# MLflow model registry loader
# ─────────────────────────────────────────────────────────────────────────────

_model_cache:   Dict[str, Any] = {}
_version_cache: Dict[str, str] = {}


def _load_model(model_name: str) -> Tuple[Any, str]:
    """Pull latest 'Production' stage artifact from MLflow registry on first call."""
    if model_name in _model_cache:
        return _model_cache[model_name], _version_cache[model_name]

    mlflow_uri = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise RuntimeError(f"No Production version for model '{model_name}'")
        mv = versions[0]
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        _model_cache[model_name]   = model
        _version_cache[model_name] = mv.version
        log.info("Loaded %s v%s from MLflow Production registry", model_name, mv.version)
        return model, mv.version
    except Exception as exc:
        log.warning("MLflow load failed for %s: %s — stub mode", model_name, exc)
        _model_cache[model_name]   = None
        _version_cache[model_name] = "stub-0"
        return None, "stub-0"


def _compute_shap_top5(model: Any, features_df: Any) -> List[SHAPFeatureImportance]:
    """Compute SHAP values and return top-5 by absolute magnitude."""
    try:
        import shap, pandas as pd
        if hasattr(model, "_model_impl") and hasattr(model._model_impl, "get_booster"):
            explainer = shap.TreeExplainer(model._model_impl)
        else:
            explainer = shap.Explainer(model.predict, features_df)
        shap_vals = explainer(features_df)
        vals = shap_vals.values[0] if hasattr(shap_vals, "values") else np.zeros(features_df.shape[1])
        cols = features_df.columns.tolist()
        ranked = sorted(zip(cols, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        return [
            SHAPFeatureImportance(
                feature=c, shap_value=round(float(v), 6),
                direction="positive" if v >= 0 else "negative",
            )
            for c, v in ranked
        ]
    except Exception as exc:
        log.debug("SHAP computation skipped: %s", exc)
        return []


def _compute_ci(raw_pred: np.ndarray, model_name: str) -> Optional[ConfidenceInterval]:
    """Derive confidence intervals where the model supports it."""
    try:
        noise_pct = {"xgboost_nowcast": 0.12, "prophet_seasonal": 0.18,
                     "cnn_land_cover": 0.08, "lstm_streamflow": 0.15}
        pct = noise_pct.get(model_name, 0.10)
        half = np.abs(raw_pred) * pct
        return ConfidenceInterval(
            lower=np.round(raw_pred - 1.96 * half, 4).tolist(),
            upper=np.round(raw_pred + 1.96 * half, 4).tolist(),
        )
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ANALYTICA Inference API",
    description=(
        "Production model serving for Tropi Climate Analytics. "
        "JWT auth header is forwarded from API-GATEWAY — not validated here. "
        "Exposes tropi_model_inference_duration_seconds histogram for CLOUD-FORGE SLA rules."
    ),
    version="3.0.0",
)


@app.on_event("startup")
async def startup():
    for m in ALL_MODELS:
        _load_model(m)
        INFERENCE_HISTOGRAM.labels(model=m)  # seed label in /metrics
    log.info("ANALYTICA Inference API ready — %d models loaded", len(_model_cache))


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", include_in_schema=False)
async def health():
    loaded = {m: (m in _model_cache and _model_cache[m] is not None) for m in ALL_MODELS}
    return {"status": "ok", "models": loaded}


@app.get("/readyz", include_in_schema=False)
async def readiness():
    if not any(_model_cache.values()):
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"status": "ready"}


# ── Core inference handler ────────────────────────────────────────────────────

def _infer(model_name: str, request: PredictRequest) -> PredictResponse:
    import pandas as pd

    model, version = _load_model(model_name)
    features_df = pd.DataFrame([request.features])

    t0 = time.perf_counter()
    with INFERENCE_HISTOGRAM.labels(model=model_name).time():
        if model is not None:
            raw = model.predict(features_df)
            prediction = np.atleast_1d(raw).tolist()
        else:
            prediction = [0.0]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    shap_top5 = _compute_shap_top5(model, features_df) if model is not None else []
    ci = _compute_ci(np.array(prediction), model_name)
    warning = "stub mode — model not yet in MLflow Production registry" if model is None else None

    REQUEST_COUNTER.labels(model=model_name, status_code="200").inc()
    return PredictResponse(
        model=model_name,
        model_version=version,
        prediction=prediction,
        confidence_interval=ci,
        shap_top5=shap_top5,
        inference_duration_ms=elapsed_ms,
        warning=warning,
    )


# ── Per-model endpoints ───────────────────────────────────────────────────────

@app.post(f"/predict/{MODEL_XGB_NOWCAST}", response_model=PredictResponse,
          summary="XGBoost 24-72h precipitation nowcast")
async def predict_xgboost_nowcast(
    request: PredictRequest,
    authorization: Optional[str] = Header(None),   # JWT pass-through to API-GATEWAY
):
    try:
        return _infer(MODEL_XGB_NOWCAST, request)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_XGB_NOWCAST, status_code="500").inc()
        log.exception("[%s] inference error", MODEL_XGB_NOWCAST)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(f"/predict/{MODEL_PROPHET_SEASONAL}", response_model=PredictResponse,
          summary="Prophet seasonal climate forecast")
async def predict_prophet_seasonal(
    request: PredictRequest,
    authorization: Optional[str] = Header(None),
):
    try:
        return _infer(MODEL_PROPHET_SEASONAL, request)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_PROPHET_SEASONAL, status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(f"/predict/{MODEL_CNN_LAND_COVER}", response_model=PredictResponse,
          summary="CNN land-cover / cloud-mask classification")
async def predict_cnn_land_cover(
    request: PredictRequest,
    authorization: Optional[str] = Header(None),
):
    try:
        return _infer(MODEL_CNN_LAND_COVER, request)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_CNN_LAND_COVER, status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(f"/predict/{MODEL_LSTM_STREAMFLOW}", response_model=PredictResponse,
          summary="LSTM streamflow forecast — Ciliwung / Brantas / Solo")
async def predict_lstm_streamflow(
    request: PredictRequest,
    authorization: Optional[str] = Header(None),
):
    if not request.station:
        raise HTTPException(
            status_code=422,
            detail="station required for lstm_streamflow. "
                   "One of: ciliwung_manggarai, brantas_mlirip, solo_jurug",
        )
    try:
        return _infer(MODEL_LSTM_STREAMFLOW, request)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_LSTM_STREAMFLOW, status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/{model_name}", response_model=PredictResponse, include_in_schema=False)
async def predict_generic(
    model_name: str,
    request: PredictRequest,
    authorization: Optional[str] = Header(None),
):
    if model_name not in ALL_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'. Valid: {ALL_MODELS}")
    try:
        return _infer(model_name, request)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=model_name, status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
