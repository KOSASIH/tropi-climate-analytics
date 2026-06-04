"""
ANALYTICA — Model Serving API
src/serving/model_server.py

FastAPI application exposing inference endpoints for all 4 ANALYTICA models.
Instruments tropi_model_inference_duration_seconds per CLOUD-FORGE SLA spec
(k8s/monitoring/README.md).

Endpoints:
  POST /predict/xgb_precip          XGBoost 24-72h precipitation nowcast
  POST /predict/prophet_seasonal    Prophet seasonal climate forecast
  POST /predict/cnn_landcover       CNN land-cover / cloud-mask classification
  POST /predict/lstm_streamflow     LSTM streamflow forecast (Ciliwung/Brantas/Solo)
  GET  /metrics                     Prometheus scrape endpoint
  GET  /healthz                     Liveness probe
  GET  /readyz                      Readiness probe

Run:
  uvicorn serving.model_server:app --host 0.0.0.0 --port 8080 --workers 2

Kubernetes deployment: k8s/serving/deployment.yaml
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port:   "8080"
    prometheus.io/path:   "/metrics"
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
)

# Local imports (resolve at runtime from /opt/analytica/src)
sys.path.insert(0, "/opt/analytica/src")

from serving.prometheus_metrics import (   # type: ignore[import]
    record_inference,
    async_inference_timer,
    INFERENCE_HISTOGRAM,
    MODEL_XGB_PRECIP,
    MODEL_PROPHET_SEASONAL,
    MODEL_CNN_LANDCOVER,
    MODEL_LSTM_STREAMFLOW,
)

log = logging.getLogger("analytica.serving")

# ─────────────────────────────────────────────────────────────────────────────
# Additional metrics
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_COUNTER = Counter(
    "tropi_model_serving_requests_total",
    "Total inference requests served",
    ["model", "status_code"],
)

MODEL_LOAD_STATE = Gauge(
    "tropi_model_loaded",
    "1 if model is loaded and healthy, 0 if not",
    ["model"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dict — keys/values depend on model. "
                    "See /docs for per-model feature schemas.",
        example={"station": "manggarai", "lookback_days": 30,
                 "rainfall_mm": [12.3, 0.0, 5.1]},
    )
    station: Optional[str] = Field(
        None,
        description="Gauge station ID (required for lstm_streamflow). "
                    "One of: ciliwung_manggarai, brantas_mlirip, solo_jurug",
    )
    horizon_hours: Optional[int] = Field(
        None,
        ge=1, le=168,
        description="Forecast horizon in hours (xgb_precip, prophet_seasonal).",
    )


class PredictResponse(BaseModel):
    model:          str
    prediction:     Any
    inference_ms:   float = Field(..., description="Inference wall-clock time (ms)")
    mlflow_run_id:  Optional[str] = None
    warning:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loader
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}


def _load_model(model_name: str) -> Any:
    if model_name in _model_cache:
        return _model_cache[model_name]

    mlflow_uri = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry   = os.getenv("ANALYTICA_MODEL_REGISTRY", "tropi-climate-models")

    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/Production"
        )
        _model_cache[model_name] = model
        MODEL_LOAD_STATE.labels(model=model_name).set(1)
        log.info(f"Loaded model: {model_name}")
        return model
    except Exception as exc:
        log.warning(f"MLflow load failed for {model_name}: {exc} — using stub")
        MODEL_LOAD_STATE.labels(model=model_name).set(0)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ANALYTICA Model Serving API",
    description = (
        "Tropi Climate Analytics inference gateway. "
        "Exposes XGBoost, Prophet, CNN and LSTM streamflow models "
        "with Prometheus instrumentation (tropi_model_inference_duration_seconds)."
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


@app.on_event("startup")
async def startup_event():
    """Warm up model cache and initialise metric labels on startup."""
    for m in [MODEL_XGB_PRECIP, MODEL_PROPHET_SEASONAL,
              MODEL_CNN_LANDCOVER, MODEL_LSTM_STREAMFLOW]:
        _load_model(m)
        # Seed histogram label so it appears in /metrics from first scrape
        INFERENCE_HISTOGRAM.labels(model=m)
    log.info("ANALYTICA serving layer ready")


# ── Metrics endpoint ─────────────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus scrape endpoint — annotated via pod annotations in k8s."""
    return Response(
        content      = generate_latest(),
        media_type   = CONTENT_TYPE_LATEST,
        status_code  = 200,
    )


# ── Health probes ─────────────────────────────────────────────────────────────

@app.get("/healthz", status_code=200, include_in_schema=False)
async def liveness():
    return {"status": "ok"}


@app.get("/readyz", status_code=200, include_in_schema=False)
async def readiness():
    loaded = {m: m in _model_cache for m in [
        MODEL_XGB_PRECIP, MODEL_PROPHET_SEASONAL,
        MODEL_CNN_LANDCOVER, MODEL_LSTM_STREAMFLOW,
    ]}
    if not any(loaded.values()):
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"status": "ready", "models": loaded}


# ── Inference helpers ─────────────────────────────────────────────────────────

def _run_inference(model_name: str, request: PredictRequest) -> Dict[str, Any]:
    """
    Synchronous inference stub — wraps the actual model predict call.
    record_inference context manager instruments the histogram.
    """
    model   = _load_model(model_name)
    t_start = time.perf_counter()

    with record_inference(model_name):
        if model is not None:
            import pandas as pd
            features_df = pd.DataFrame([request.features])
            raw_pred    = model.predict(features_df)
            prediction  = raw_pred.tolist() if hasattr(raw_pred, "tolist") else raw_pred
        else:
            # Stub fallback when model not yet promoted to Production in MLflow
            prediction = {"stub": True, "model": model_name,
                          "note": "model not yet in MLflow Production registry"}

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return {"prediction": prediction, "inference_ms": round(elapsed_ms, 3)}


# ── Per-model endpoints ───────────────────────────────────────────────────────

@app.post(
    f"/predict/{MODEL_XGB_PRECIP}",
    response_model  = PredictResponse,
    summary         = "XGBoost 24-72h precipitation nowcast",
)
async def predict_xgb_precip(request: PredictRequest):
    try:
        result = _run_inference(MODEL_XGB_PRECIP, request)
        REQUEST_COUNTER.labels(model=MODEL_XGB_PRECIP, status_code="200").inc()
        return PredictResponse(model=MODEL_XGB_PRECIP, **result)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_XGB_PRECIP, status_code="500").inc()
        log.exception(f"[{MODEL_XGB_PRECIP}] Inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    f"/predict/{MODEL_PROPHET_SEASONAL}",
    response_model  = PredictResponse,
    summary         = "Prophet seasonal climate forecast",
)
async def predict_prophet_seasonal(request: PredictRequest):
    try:
        result = _run_inference(MODEL_PROPHET_SEASONAL, request)
        REQUEST_COUNTER.labels(model=MODEL_PROPHET_SEASONAL, status_code="200").inc()
        return PredictResponse(model=MODEL_PROPHET_SEASONAL, **result)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_PROPHET_SEASONAL, status_code="500").inc()
        log.exception(f"[{MODEL_PROPHET_SEASONAL}] Inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    f"/predict/{MODEL_CNN_LANDCOVER}",
    response_model  = PredictResponse,
    summary         = "CNN land-cover / cloud-mask classification",
)
async def predict_cnn_landcover(request: PredictRequest):
    try:
        result = _run_inference(MODEL_CNN_LANDCOVER, request)
        REQUEST_COUNTER.labels(model=MODEL_CNN_LANDCOVER, status_code="200").inc()
        return PredictResponse(model=MODEL_CNN_LANDCOVER, **result)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_CNN_LANDCOVER, status_code="500").inc()
        log.exception(f"[{MODEL_CNN_LANDCOVER}] Inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    f"/predict/{MODEL_LSTM_STREAMFLOW}",
    response_model  = PredictResponse,
    summary         = "LSTM streamflow forecast (Ciliwung / Brantas / Solo)",
)
async def predict_lstm_streamflow(request: PredictRequest):
    if not request.station:
        raise HTTPException(
            status_code = 422,
            detail      = "station required for lstm_streamflow. "
                          "One of: ciliwung_manggarai, brantas_mlirip, solo_jurug",
        )
    try:
        result = _run_inference(MODEL_LSTM_STREAMFLOW, request)
        REQUEST_COUNTER.labels(model=MODEL_LSTM_STREAMFLOW, status_code="200").inc()
        return PredictResponse(model=MODEL_LSTM_STREAMFLOW, **result)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=MODEL_LSTM_STREAMFLOW, status_code="500").inc()
        log.exception(f"[{MODEL_LSTM_STREAMFLOW}] Inference error")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Generic passthrough endpoint ──────────────────────────────────────────────

@app.post(
    "/predict/{model_name}",
    response_model  = PredictResponse,
    summary         = "Generic inference passthrough",
    include_in_schema = False,
)
async def predict_generic(model_name: str, request: PredictRequest):
    valid = [MODEL_XGB_PRECIP, MODEL_PROPHET_SEASONAL,
             MODEL_CNN_LANDCOVER, MODEL_LSTM_STREAMFLOW]
    if model_name not in valid:
        raise HTTPException(
            status_code = 404,
            detail      = f"Unknown model '{model_name}'. Valid: {valid}",
        )
    try:
        result = _run_inference(model_name, request)
        REQUEST_COUNTER.labels(model=model_name, status_code="200").inc()
        return PredictResponse(model=model_name, **result)
    except Exception as exc:
        REQUEST_COUNTER.labels(model=model_name, status_code="500").inc()
        log.exception(f"[{model_name}] Inference error")
        raise HTTPException(status_code=500, detail=str(exc))
