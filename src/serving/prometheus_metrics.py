"""
ANALYTICA — Prometheus Metrics
src/serving/prometheus_metrics.py

Defines tropi_model_inference_duration_seconds histogram (CLOUD-FORGE spec).
Drives:
  TropiModelInferenceP99High   p99 > 5s  → warning
  TropiModelInferenceP99Critical p99 > 15s → critical

Usage (FastAPI middleware):
    from serving.prometheus_metrics import INFERENCE_HISTOGRAM, record_inference
    with record_inference("xgb_precip"):
        result = model.predict(features)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Histogram, REGISTRY, CollectorRegistry

# ─────────────────────────────────────────────────────────────────────────────
# Model name constants  (must match Alertmanager label selectors)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_XGB_PRECIP       = "xgb_precip"
MODEL_PROPHET_SEASONAL = "prophet_seasonal"
MODEL_CNN_LANDCOVER    = "cnn_landcover"
MODEL_LSTM_STREAMFLOW  = "lstm_streamflow"

ALL_MODELS = [
    MODEL_XGB_PRECIP,
    MODEL_PROPHET_SEASONAL,
    MODEL_CNN_LANDCOVER,
    MODEL_LSTM_STREAMFLOW,
]

# ─────────────────────────────────────────────────────────────────────────────
# Histogram definition
# Buckets span sub-second (fast batch) → 20s (slow CNN) with explicit 5s / 15s
# boundaries that match the TropiModelInferenceP99High / Critical thresholds.
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_BUCKETS = (
    0.05, 0.1, 0.25, 0.5,
    1.0, 2.0, 3.0,
    5.0,          # TropiModelInferenceP99High threshold
    7.5, 10.0, 12.5,
    15.0,         # TropiModelInferenceP99Critical threshold
    20.0, 30.0, float("inf"),
)

INFERENCE_HISTOGRAM: Histogram = Histogram(
    name        = "tropi_model_inference_duration_seconds",
    documentation = (
        "ANALYTICA model inference latency in seconds. "
        "Histogram with label model=<model_name>. "
        "Drives TropiModelInferenceP99High (p99>5s) and "
        "TropiModelInferenceP99Critical (p99>15s) SLA alert rules."
    ),
    labelnames  = ["model"],
    buckets     = INFERENCE_BUCKETS,
)


@contextmanager
def record_inference(model_name: str) -> Generator[None, None, None]:
    """
    Context manager — wraps a model inference call and records duration.

    Example:
        with record_inference("xgb_precip"):
            preds = xgb_model.predict(X)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        INFERENCE_HISTOGRAM.labels(model=model_name).observe(elapsed)


def inference_timer(model_name: str):
    """
    Decorator variant for synchronous inference functions.

    Example:
        @inference_timer("prophet_seasonal")
        def predict_seasonal(features): ...
    """
    def decorator(fn):
        import functools
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with record_inference(model_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def async_inference_timer(model_name: str):
    """
    Decorator variant for async inference functions (FastAPI endpoints).

    Example:
        @app.post("/predict/cnn_landcover")
        @async_inference_timer("cnn_landcover")
        async def predict_cnn(request: PredictRequest): ...
    """
    def decorator(fn):
        import asyncio, functools
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                INFERENCE_HISTOGRAM.labels(model=model_name).observe(elapsed)
        return wrapper
    return decorator
