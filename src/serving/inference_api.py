"""
ANALYTICA Inference API — Sprint 6 H1
FastAPI router for all 5 ANALYTICA prediction endpoints.

Mount: add to src/api/main.py alongside the HYDROLOGIS router:
    from src.serving.inference_api import router as inference_router
    app.include_router(inference_router)

Per-request flow (MISS path):
    InferenceCache.get()
    -> MISS: FeatureStoreClient.get_online_features()
             -> model.predict()
             -> SHAPExplainer (top-5 features, async sidecar write)
             -> InferenceCache.set()
             -> return
    -> HIT:  return cached (cache_hit=True)

Endpoints:
    POST /predict/precipitation  -> XGBoostPrecipModel
    POST /predict/seasonal       -> ProphetSeasonalModel
    POST /predict/landcover      -> CNNLandCoverModel
    POST /predict/streamflow     -> LSTMStreamflowModel
    POST /predict/climate        -> TFTClimateModel
    GET  /healthz/inference      -> subsystem health
    GET  /metrics                -> Prometheus scrape (delegated to prometheus_client)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timezone
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus (Sprint 6 metrics.py exports)
# ---------------------------------------------------------------------------
try:
    from src.data.metrics import INFERENCE_LATENCY, INFERENCE_REQUESTS
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    logger.warning("metrics.py not importable — Prometheus counters disabled")

# ---------------------------------------------------------------------------
# Dependency registry (lazy singletons)
# ---------------------------------------------------------------------------
_cache: Any              = None
_feature_store: Any      = None
_shap_explainer: Any     = None
_models: dict[str, Any]  = {}


def _get_cache():
    global _cache
    if _cache is None:
        from src.serving.inference_cache import InferenceCache
        _cache = InferenceCache()
    return _cache


def _get_feature_store():
    global _feature_store
    if _feature_store is None:
        from src.data.feature_store_client import FeatureStoreClient
        _feature_store = FeatureStoreClient()
    return _feature_store


def _get_shap():
    global _shap_explainer
    if _shap_explainer is None:
        from src.explainability.shap_explainer import SHAPExplainer
        _shap_explainer = SHAPExplainer()
    return _shap_explainer


def _get_model(model_id: str) -> Any:
    if model_id not in _models:
        try:
            from src.training.mlflow_registry import MLflowRegistry
            import mlflow.pyfunc
            MLflowRegistry().get_latest_production(model_id)
            _models[model_id] = mlflow.pyfunc.load_model(f"models:/{model_id}/Production")
        except Exception as exc:
            logger.warning("Model %s unavailable (%s) — using stub", model_id, exc)
            _models[model_id] = _ModelStub(model_id)
    return _models[model_id]


class _ModelStub:
    """Null model stub — returns None prediction when registry is unavailable."""
    FEATURE_REFS = []
    version = "stub"

    def __init__(self, model_id: str):
        self.model_id = model_id

    def predict(self, **kwargs):
        return {"prediction": None, "confidence_interval": None}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    entity_id:        str            = Field(..., description="Station or entity identifier")
    grid_cell_id:     str            = Field(..., description="4km grid cell ID (e.g. gc_0001)")
    forecast_horizon: int            = Field(..., ge=1, le=720, description="Hours ahead to forecast")
    issued_date:      date           = Field(default_factory=date.today)


class SHAPEntry(BaseModel):
    feature:    str
    shap_value: float
    rank:       int


class PredictResponse(BaseModel):
    model_id:               str
    prediction:             Any
    confidence_interval:    Optional[Any]
    feature_importance_top5: List[SHAPEntry]
    cache_hit:              bool
    latency_ms:             float


class HealthResponse(BaseModel):
    status:                   str
    models_loaded:            int
    cache_available:          bool
    feature_store_available:  bool
    checked_at:               str


# ---------------------------------------------------------------------------
# Core inference logic
# ---------------------------------------------------------------------------

async def _run_inference(
    model_id:      str,
    model_name:    str,
    request:       PredictRequest,
    endpoint_slug: str,
) -> PredictResponse:
    """
    Shared inference pathway for all 5 prediction endpoints.
    Implements cache-aside with SHAP explainability on MISS path.
    """
    t0         = time.perf_counter()
    cache      = _get_cache()
    fs         = _get_feature_store()
    shap       = _get_shap()
    model      = _get_model(model_id)

    input_key  = {
        "entity_id":        request.entity_id,
        "grid_cell_id":     request.grid_cell_id,
        "forecast_horizon": request.forecast_horizon,
        "issued_date":      request.issued_date.isoformat(),
    }
    model_version = getattr(model, "version", "prod")

    # --- Cache check ---
    cached = cache.get(model_id, model_version, input_key)
    if cached:
        latency_ms = (time.perf_counter() - t0) * 1000
        if _METRICS_AVAILABLE:
            INFERENCE_REQUESTS.labels(model_id=model_id, status="cache_hit").inc()
            INFERENCE_LATENCY.labels(model_id=model_id, endpoint=endpoint_slug).observe(latency_ms / 1000)
        return PredictResponse(
            model_id=model_id,
            prediction=cached.get("prediction"),
            confidence_interval=cached.get("confidence_interval"),
            feature_importance_top5=cached.get("feature_importance_top5", []),
            cache_hit=True,
            latency_ms=round(latency_ms, 2),
        )

    # --- MISS path ---
    try:
        features_df = fs.get_online_features(
            entity_rows=[{"grid_cell_id": request.grid_cell_id, "entity_id": request.entity_id}],
            feature_refs=getattr(model, "FEATURE_REFS", []),
        )

        raw_pred = model.predict(
            features=features_df,
            forecast_horizon=request.forecast_horizon,
            issued_date=request.issued_date,
        )

        # SHAP explainability — top-5 features (async sidecar write)
        shap_entries: List[SHAPEntry] = []
        try:
            shap_raw = _get_shap_top5(shap, model_id, model, features_df, request)
            shap_entries = shap_raw
            # Fire-and-forget sidecar write
            asyncio.create_task(_write_shap_sidecar(shap_raw, model_id, request))
        except Exception as shap_exc:
            logger.warning("SHAP failed for %s/%s: %s", model_id, request.grid_cell_id, shap_exc)

        result_payload = {
            "prediction":             raw_pred.get("prediction"),
            "confidence_interval":    raw_pred.get("confidence_interval"),
            "feature_importance_top5": [e.dict() for e in shap_entries],
        }
        cache.set(model_id, model_version, input_key, result_payload)

        latency_ms = (time.perf_counter() - t0) * 1000
        if _METRICS_AVAILABLE:
            INFERENCE_REQUESTS.labels(model_id=model_id, status="success").inc()
            INFERENCE_LATENCY.labels(model_id=model_id, endpoint=endpoint_slug).observe(latency_ms / 1000)

        return PredictResponse(
            model_id=model_id,
            prediction=result_payload["prediction"],
            confidence_interval=result_payload["confidence_interval"],
            feature_importance_top5=shap_entries,
            cache_hit=False,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        if _METRICS_AVAILABLE:
            INFERENCE_REQUESTS.labels(model_id=model_id, status="error").inc()
            INFERENCE_LATENCY.labels(model_id=model_id, endpoint=endpoint_slug).observe(latency_ms / 1000)
        logger.exception("Inference error for %s: %s", model_id, exc)
        raise HTTPException(status_code=500, detail=f"Inference failed for {model_id}: {exc}")


def _get_shap_top5(
    shap, model_id: str, model, features_df: pd.DataFrame, request: PredictRequest
) -> List[SHAPEntry]:
    """Dispatch to the correct SHAPExplainer method; return top-5 by abs(shap_value)."""
    if "xgb" in model_id:
        raw = shap.explain_xgb(model, features_df)
    elif "cnn" in model_id:
        X_arr = features_df.values if hasattr(features_df, "values") else np.array(features_df)
        raw = shap.explain_cnn(model, X_arr)
    else:
        raw = shap.explain_deep(model, features_df)

    sorted_shap = sorted(raw, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)[:5]
    return [
        SHAPEntry(feature=e["feature"], shap_value=e["shap_value"], rank=i + 1)
        for i, e in enumerate(sorted_shap)
    ]


async def _write_shap_sidecar(
    shap_entries: List[SHAPEntry], model_id: str, request: PredictRequest
) -> None:
    """Async sidecar: persist SHAP output to workspace/output/explainability/."""
    import json
    from pathlib import Path

    out_dir = Path("workspace/output/explainability")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model_id}_{request.grid_cell_id}_{request.issued_date.isoformat()}.json"
    out_path = out_dir / fname
    payload = {
        "model_id":      model_id,
        "grid_cell_id":  request.grid_cell_id,
        "issued_date":   request.issued_date.isoformat(),
        "shap_values":   [e.dict() for e in shap_entries],
        "written_at":    datetime.now(timezone.utc).isoformat(),
    }
    await asyncio.to_thread(out_path.write_text, json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/predict", tags=["inference"])


@router.post("/precipitation", response_model=PredictResponse, summary="XGBoost precipitation nowcast")
async def predict_precipitation(request: PredictRequest):
    return await _run_inference("xgb_precip_nowcast", "XGBoostPrecipModel", request, "precipitation")


@router.post("/seasonal", response_model=PredictResponse, summary="Prophet seasonal climate forecast")
async def predict_seasonal(request: PredictRequest):
    return await _run_inference("prophet_seasonal", "ProphetSeasonalModel", request, "seasonal")


@router.post("/landcover", response_model=PredictResponse, summary="CNN land cover classification")
async def predict_landcover(request: PredictRequest):
    return await _run_inference("cnn_landcover", "CNNLandCoverModel", request, "landcover")


@router.post("/streamflow", response_model=PredictResponse, summary="LSTM streamflow forecast")
async def predict_streamflow(request: PredictRequest):
    return await _run_inference("lstm_streamflow", "LSTMStreamflowModel", request, "streamflow")


@router.post("/climate", response_model=PredictResponse, summary="TFT multi-variate climate forecast")
async def predict_climate(request: PredictRequest):
    return await _run_inference("tft_climate_forecast", "TFTClimateModel", request, "climate")


@router.get("/healthz/inference", response_model=HealthResponse, tags=["health"],
            summary="ANALYTICA inference subsystem health")
async def healthz_inference():
    cache_ok = False
    fs_ok    = False
    try:
        _get_cache().ping()
        cache_ok = True
    except Exception:
        pass
    try:
        _get_feature_store().ping()
        fs_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        models_loaded=len(_models),
        cache_available=cache_ok,
        feature_store_available=fs_ok,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )
