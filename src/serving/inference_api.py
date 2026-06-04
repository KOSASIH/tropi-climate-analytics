"""
Inference API — ANALYTICA Sprint 5 F1
FastAPI serving layer for all 5 ANALYTICA predictive models.

Endpoints:
  POST /predict/precipitation  → XGBoostPrecipModel
  POST /predict/seasonal       → ProphetSeasonalModel
  POST /predict/landcover      → CNNLandCoverModel
  POST /predict/streamflow     → LSTMStreamflowModel
  POST /predict/climate        → TFTClimateModel

Request flow: InferenceCache.get() → HIT: return cached
                                   → MISS: FeatureStoreClient.get_features()
                                           → model.predict()
                                           → InferenceCache.set()
                                           → return

Prometheus:
  tropi_inference_latency_seconds{model_id, endpoint}      ← Histogram
  tropi_inference_requests_total{model_id, status}          ← Counter
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data.metrics import INFERENCE_LATENCY, INFERENCE_REQUESTS
from src.serving.inference_cache import InferenceCache
from src.data.feature_store_client import FeatureStoreClient
from src.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ANALYTICA Inference API",
    version="1.0.0",
    description="Multi-model climate prediction endpoints with cache-aside and SHAP explainability.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Shared singletons (initialised at startup)
# ---------------------------------------------------------------------------
_CACHE: InferenceCache
_FEATURE_STORE: FeatureStoreClient
_SHAP: SHAPExplainer
_MODEL_REGISTRY: Dict[str, Any] = {}
_MODELS_LOADED: int = 0


@app.on_event("startup")
async def startup() -> None:
    global _CACHE, _FEATURE_STORE, _SHAP, _MODEL_REGISTRY, _MODELS_LOADED
    _CACHE = InferenceCache()
    _FEATURE_STORE = FeatureStoreClient()
    _SHAP = SHAPExplainer()
    _MODEL_REGISTRY = _load_models()
    _MODELS_LOADED = len(_MODEL_REGISTRY)
    logger.info("ANALYTICA Inference API ready — %d models loaded", _MODELS_LOADED)


def _load_models() -> Dict[str, Any]:
    """Lazy-load all registered ANALYTICA models. Returns empty stubs if not available."""
    registry: Dict[str, Any] = {}
    model_loaders = {
        "xgb_precip":   "src.models.xgboost_nowcast.XGBoostPrecipModel",
        "prophet":       "src.models.prophet_seasonal.ProphetSeasonalModel",
        "cnn_landcover": "src.models.cnn_land_cover.CNNLandCoverModel",
        "lstm_stream":   "src.models.lstm_streamflow.LSTMStreamflowModel",
        "tft_climate":   "src.models.climate_transformer.TFTClimateModel",
    }
    for model_id, dotpath in model_loaders.items():
        try:
            module_path, class_name = dotpath.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            registry[model_id] = cls.load_from_registry()
            logger.info("Loaded model: %s", model_id)
        except Exception as exc:
            logger.warning("Model %s unavailable (%s) — stub installed", model_id, exc)
            registry[model_id] = _ModelStub(model_id)
    return registry


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    entity_id:        str        = Field(..., description="Entity identifier (e.g. BMKG station ID)")
    grid_cell_id:     str        = Field(..., description="Grid cell ID (4km resolution)")
    forecast_horizon: int        = Field(..., ge=1, le=720, description="Forecast horizon in hours")
    issued_date:      date       = Field(..., description="Date for which forecast is issued")


class PredictionResponse(BaseModel):
    model_id:              str
    prediction:            Any
    confidence_interval:   Optional[Dict[str, float]]
    feature_importance_top5: List[Dict[str, Any]]
    cache_hit:             bool
    latency_ms:            float


class HealthResponse(BaseModel):
    status:                  str
    models_loaded:           int
    cache_available:         bool
    feature_store_available: bool


# ---------------------------------------------------------------------------
# Core inference helper
# ---------------------------------------------------------------------------

def _run_inference(
    model_id: str,
    endpoint: str,
    request: PredictionRequest,
) -> PredictionResponse:
    t0 = time.perf_counter()
    model = _MODEL_REGISTRY.get(model_id)
    if model is None:
        INFERENCE_REQUESTS.labels(model_id=model_id, status="error").inc()
        raise HTTPException(status_code=503, detail=f"Model {model_id} not loaded")

    cache_key_input = request.model_dump()
    model_version = getattr(model, "version", "latest")

    # --- Cache check ---
    cached = _CACHE.get(model_id, model_version, cache_key_input)
    if cached is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        INFERENCE_REQUESTS.labels(model_id=model_id, status="cache_hit").inc()
        INFERENCE_LATENCY.labels(model_id=model_id, endpoint=endpoint).observe(latency_ms / 1000)
        return PredictionResponse(
            model_id=model_id,
            prediction=cached["prediction"],
            confidence_interval=cached.get("confidence_interval"),
            feature_importance_top5=cached.get("feature_importance_top5", []),
            cache_hit=True,
            latency_ms=round(latency_ms, 2),
        )

    # --- Feature retrieval ---
    try:
        features_df = _FEATURE_STORE.get_online_features(
            entity_rows=[{
                "grid_cell_id": request.grid_cell_id,
                "station_id":   request.entity_id,
            }],
            feature_refs=getattr(model, "FEATURE_REFS", []),
        )
    except Exception as exc:
        logger.warning("Feature store unavailable for %s: %s — using empty features", model_id, exc)
        import pandas as pd
        features_df = pd.DataFrame()

    # --- Inference ---
    try:
        raw_result = model.predict(
            features=features_df,
            forecast_horizon=request.forecast_horizon,
            issued_date=request.issued_date,
        )
    except Exception as exc:
        INFERENCE_REQUESTS.labels(model_id=model_id, status="error").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    prediction       = raw_result.get("prediction")
    conf_interval    = raw_result.get("confidence_interval")

    # --- SHAP explainability (top-5 subset of top-10) ---
    try:
        shap_result = _SHAP.explain_for_model(
            model_id=model_id,
            model=model,
            features=features_df,
            grid_cell_id=request.grid_cell_id,
            issued_date=str(request.issued_date),
        )
        importance_top5 = sorted(shap_result, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
    except Exception as exc:
        logger.warning("SHAP unavailable for %s: %s", model_id, exc)
        importance_top5 = []

    # --- Store in cache ---
    payload = {
        "prediction":            prediction,
        "confidence_interval":   conf_interval,
        "feature_importance_top5": importance_top5,
    }
    _CACHE.set(model_id, model_version, cache_key_input, payload)

    latency_ms = (time.perf_counter() - t0) * 1000
    INFERENCE_REQUESTS.labels(model_id=model_id, status="success").inc()
    INFERENCE_LATENCY.labels(model_id=model_id, endpoint=endpoint).observe(latency_ms / 1000)

    return PredictionResponse(
        model_id=model_id,
        prediction=prediction,
        confidence_interval=conf_interval,
        feature_importance_top5=importance_top5,
        cache_hit=False,
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@app.post("/predict/precipitation", response_model=PredictionResponse, tags=["Inference"])
async def predict_precipitation(request: PredictionRequest):
    """XGBoost 24–72 h precipitation nowcast."""
    return _run_inference("xgb_precip", "/predict/precipitation", request)


@app.post("/predict/seasonal", response_model=PredictionResponse, tags=["Inference"])
async def predict_seasonal(request: PredictionRequest):
    """Prophet seasonal climate forecast."""
    return _run_inference("prophet", "/predict/seasonal", request)


@app.post("/predict/landcover", response_model=PredictionResponse, tags=["Inference"])
async def predict_landcover(request: PredictionRequest):
    """CNN land cover classification from satellite imagery."""
    return _run_inference("cnn_landcover", "/predict/landcover", request)


@app.post("/predict/streamflow", response_model=PredictionResponse, tags=["Inference"])
async def predict_streamflow(request: PredictionRequest):
    """LSTM streamflow / flood early warning forecast."""
    return _run_inference("lstm_stream", "/predict/streamflow", request)


@app.post("/predict/climate", response_model=PredictionResponse, tags=["Inference"])
async def predict_climate(request: PredictionRequest):
    """TFT multi-variate climate forecast (T2M, RH, wind)."""
    return _run_inference("tft_climate", "/predict/climate", request)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz():
    cache_ok   = _CACHE._redis is not None
    fs_ok      = _FEATURE_STORE._store is not None
    return HealthResponse(
        status="ok",
        models_loaded=_MODELS_LOADED,
        cache_available=cache_ok,
        feature_store_available=fs_ok,
    )


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint (delegated to prometheus_client)
# ---------------------------------------------------------------------------

@app.get("/metrics", tags=["Observability"], include_in_schema=False)
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

class _ModelStub:
    """Null-inference stub when a model class is unavailable at load time."""
    FEATURE_REFS = []
    version = "stub"

    def __init__(self, model_id: str):
        self.model_id = model_id

    def predict(self, features, forecast_horizon, issued_date):
        return {"prediction": None, "confidence_interval": None}

    @classmethod
    def load_from_registry(cls):
        return cls("stub")
