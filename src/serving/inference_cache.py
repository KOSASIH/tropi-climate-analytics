"""
Inference Cache — ANALYTICA Sprint 5 D2
Redis-backed cache for model server inference results.

Key schema: {model_name}:{model_version}:{input_hash_sha256}
TTL policy:
  XGBoost  1800 s | Prophet 3600 s | CNN 7200 s | LSTM 1800 s | TFT 3600 s

Pattern: cache-aside
  GET key → hit: return cached prediction
           → miss: run inference → SET key with TTL → return prediction

Prometheus:
  tropi_inference_cache_hit_total{model_name}
  tropi_inference_cache_miss_total{model_name}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from prometheus_client import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
_CACHE_HIT = Counter(
    "tropi_inference_cache_hit_total",
    "Inference cache hits by model name",
    ["model_name"],
)
_CACHE_MISS = Counter(
    "tropi_inference_cache_miss_total",
    "Inference cache misses by model name",
    ["model_name"],
)

# ---------------------------------------------------------------------------
# TTL policy
# ---------------------------------------------------------------------------
TTL_POLICY: Dict[str, int] = {
    "xgb":     1800,  # XGBoost precipitation nowcast
    "prophet": 3600,  # Prophet seasonal forecast
    "cnn":     7200,  # CNN land cover classifier
    "lstm":    1800,  # LSTM streamflow forecast
    "tft":     3600,  # TFT multi-variate climate
}
_DEFAULT_TTL = 1800

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _resolve_ttl(model_name: str) -> int:
    for prefix, ttl in TTL_POLICY.items():
        if model_name.lower().startswith(prefix):
            return ttl
    return _DEFAULT_TTL


def _input_hash(input_data: Any) -> str:
    payload = json.dumps(input_data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _make_key(model_name: str, model_version: str, input_data: Any) -> str:
    return f"{model_name}:{model_version}:{_input_hash(input_data)}"


# ---------------------------------------------------------------------------
# InferenceCache
# ---------------------------------------------------------------------------

class InferenceCache:
    """
    Redis-backed inference cache using cache-aside pattern.

    Usage::

        cache = InferenceCache()

        result = cache.predict(
            model_name="xgb_precip_nowcast",
            model_version="2.1",
            input_data=feature_dict,
            inference_fn=lambda: model.predict(feature_dict),
        )
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._url = redis_url or REDIS_URL
        self._redis = self._connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        model_name: str,
        model_version: str,
        input_data: Any,
        inference_fn: Callable[[], Any],
    ) -> Any:
        """
        Cache-aside inference wrapper.

        1. Compute cache key from (model_name, model_version, sha256(input_data)).
        2. Redis GET → hit: deserialise and return.
        3. Miss: call inference_fn(), store result in Redis, return result.
        """
        key = _make_key(model_name, model_version, input_data)

        # --- Cache check ---
        cached = self._get(key, model_name)
        if cached is not None:
            return cached

        # --- Cache miss: run inference ---
        result = inference_fn()

        # --- Store result ---
        self._set(key, result, model_name)
        return result

    def get(self, model_name: str, model_version: str, input_data: Any) -> Optional[Any]:
        """Explicit GET — returns None on miss or bypass."""
        key = _make_key(model_name, model_version, input_data)
        return self._get(key, model_name)

    def set(
        self,
        model_name: str,
        model_version: str,
        input_data: Any,
        result: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Explicit SET."""
        key = _make_key(model_name, model_version, input_data)
        self._set(key, result, model_name, ttl_seconds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, key: str, model_name: str) -> Optional[Any]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            if raw is None:
                _CACHE_MISS.labels(model_name=model_name).inc()
                return None
            _CACHE_HIT.labels(model_name=model_name).inc()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("InferenceCache.get failed (key=%s): %s", key, exc)
            _CACHE_MISS.labels(model_name=model_name).inc()
            return None

    def _set(
        self,
        key: str,
        result: Any,
        model_name: str,
        ttl: Optional[int] = None,
    ) -> None:
        if self._redis is None:
            return
        ttl = ttl if ttl is not None else _resolve_ttl(model_name)
        try:
            self._redis.setex(key, ttl, json.dumps(result, default=str))
        except Exception as exc:
            logger.warning("InferenceCache.set failed (key=%s): %s", key, exc)

    def _connect(self):
        try:
            import redis  # type: ignore
            client = redis.from_url(
                self._url,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            logger.info("InferenceCache: connected to Redis at %s", self._url)
            return client
        except Exception as exc:
            logger.warning(
                "InferenceCache: Redis unavailable (%s) — bypass mode active. "
                "Inference will proceed uncached.",
                exc,
            )
            return None
