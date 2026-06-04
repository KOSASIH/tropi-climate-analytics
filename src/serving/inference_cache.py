"""
Inference Cache — ANALYTICA Sprint 5 D4
Redis-backed cache for high-frequency grid cell inference results.
Bypasses gracefully if Redis is unavailable — inference is never blocked.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from prometheus_client import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
_CACHE_OPS = Counter(
    "tropi_inference_cache_ops_total",
    "Total inference cache operations by model, op type, and result",
    ["model_id", "op", "result"],  # op: get|set  result: hit|miss|bypass
)

# ---------------------------------------------------------------------------
# TTL policy (seconds) keyed on model_id prefix / alias
# ---------------------------------------------------------------------------
TTL_POLICY: dict[str, int] = {
    "xgb_precip":          30 * 60,    # 30 min  — precipitation nowcast
    "prophet_seasonal":     6 * 3600,  # 6 h     — seasonal climate forecast
    "cnn_landcover":       24 * 3600,  # 24 h    — land cover classification
    "lstm_streamflow":      1 * 3600,  # 1 h     — streamflow forecast
    "tft_climate":          2 * 3600,  # 2 h     — TFT multi-variate climate
}
_DEFAULT_TTL = 3600


def _resolve_ttl(model_id: str) -> int:
    for prefix, ttl in TTL_POLICY.items():
        if model_id.startswith(prefix):
            return ttl
    return _DEFAULT_TTL


def _build_cache_key(
    model_id: str,
    grid_cell_id: str,
    forecast_horizon: str,
    issued_date: str,
) -> str:
    """Canonical key: {model_id}:{grid_cell_id}:{forecast_horizon}:{issued_date}"""
    return f"{model_id}:{grid_cell_id}:{forecast_horizon}:{issued_date}"


# ---------------------------------------------------------------------------
# InferenceCache
# ---------------------------------------------------------------------------
class InferenceCache:
    """
    Redis-backed inference result cache.

    On Redis unavailable: logs WARNING and operates as a no-op pass-through.
    Inference is never blocked or degraded by cache failures.

    Usage::

        cache = InferenceCache()
        key = cache.build_key("xgb_precip", "grid_001", "24h", "2026-06-04")

        result = cache.get(key, model_id="xgb_precip")
        if result is None:
            result = model.predict(features)
            cache.set(key, result, model_id="xgb_precip")
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis = self._connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def build_key(
        model_id: str,
        grid_cell_id: str,
        forecast_horizon: str,
        issued_date: str,
    ) -> str:
        return _build_cache_key(model_id, grid_cell_id, forecast_horizon, issued_date)

    def get(self, cache_key: str, model_id: str = "unknown") -> Optional[dict]:
        """
        Retrieve a cached inference result.

        Returns the deserialized dict if present, or None on miss/bypass.
        """
        if self._redis is None:
            _CACHE_OPS.labels(model_id=model_id, op="get", result="bypass").inc()
            return None

        try:
            raw = self._redis.get(cache_key)
            if raw is None:
                _CACHE_OPS.labels(model_id=model_id, op="get", result="miss").inc()
                return None
            _CACHE_OPS.labels(model_id=model_id, op="get", result="hit").inc()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("InferenceCache.get failed (key=%s): %s", cache_key, exc)
            _CACHE_OPS.labels(model_id=model_id, op="get", result="bypass").inc()
            return None

    def set(
        self,
        cache_key: str,
        result: dict,
        model_id: str = "unknown",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store an inference result.

        ttl_seconds defaults to the model-specific TTL policy if not provided.
        """
        if self._redis is None:
            _CACHE_OPS.labels(model_id=model_id, op="set", result="bypass").inc()
            return

        ttl = ttl_seconds if ttl_seconds is not None else _resolve_ttl(model_id)
        try:
            self._redis.setex(cache_key, ttl, json.dumps(result, default=str))
            _CACHE_OPS.labels(model_id=model_id, op="set", result="hit").inc()
        except Exception as exc:
            logger.warning("InferenceCache.set failed (key=%s): %s", cache_key, exc)
            _CACHE_OPS.labels(model_id=model_id, op="set", result="bypass").inc()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self):
        """Attempt Redis connection; return None on failure (bypass mode)."""
        try:
            import redis  # type: ignore

            client = redis.from_url(self._url, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
            logger.info("InferenceCache: connected to Redis at %s", self._url)
            return client
        except Exception as exc:
            logger.warning(
                "InferenceCache: Redis unavailable (%s) — running in bypass mode. "
                "Inference will proceed without caching.",
                exc,
            )
            return None
