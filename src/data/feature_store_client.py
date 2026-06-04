"""
Feature Store Client — ANALYTICA Sprint 5 D1
Wraps Feast online/offline store for ANALYTICA model serving and training.

Supports entity types: station_id, grid_cell_id, watershed_id
Online path: Redis-cached (TTL 300 s) → Feast online store
Historical path: Feast offline store (no cache)

Prometheus:
  tropi_feature_store_cache_hit_total{entity_type}
  tropi_feature_store_latency_seconds{method}  ← Histogram
MLflow: tags each training run with feature_store_version
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
_CACHE_HITS = Counter(
    "tropi_feature_store_cache_hit_total",
    "Feature store Redis cache hits by entity type",
    ["entity_type"],
)

_LATENCY = Histogram(
    "tropi_feature_store_latency_seconds",
    "Feature store call latency by method",
    ["method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Entity type registry
# ---------------------------------------------------------------------------
ENTITY_TYPES = {"station_id", "grid_cell_id", "watershed_id"}
ONLINE_CACHE_TTL = 300  # seconds — online features only
FEATURE_STORE_VERSION = os.environ.get("FEATURE_STORE_VERSION", "v1.0")
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "workspace/feature_store/feature_repo")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/1")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entity_type_from_rows(entity_rows: List[Dict[str, Any]]) -> str:
    """Infer entity type from entity_rows key set."""
    if not entity_rows:
        return "unknown"
    keys = set(entity_rows[0].keys())
    for etype in ENTITY_TYPES:
        if etype in keys:
            return etype
    return "unknown"


def _redis_key(entity_rows: List[Dict[str, Any]], feature_refs: List[str]) -> str:
    payload = json.dumps({"entities": entity_rows, "features": sorted(feature_refs)},
                         sort_keys=True, default=str)
    return "tropi:fs:" + hashlib.sha256(payload.encode()).hexdigest()[:32]


def _connect_redis():
    try:
        import redis  # type: ignore
        client = redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        logger.info("FeatureStoreClient: Redis connected at %s", REDIS_URL)
        return client
    except Exception as exc:
        logger.warning("FeatureStoreClient: Redis unavailable (%s) — online cache disabled", exc)
        return None


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class FeatureStoreClient:
    """
    Feast-backed feature store client for ANALYTICA model serving and training.

    Usage — online (low-latency, cached)::

        client = FeatureStoreClient()
        df = client.get_online_features(
            entity_rows=[{"station_id": "bmkg_5201"}, {"station_id": "bmkg_1102"}],
            feature_refs=["weather_stats:t2m_mean_24h", "weather_stats:rh_mean_24h"],
        )

    Usage — historical (training, uncached)::

        df = client.get_historical_features(
            entity_df=entity_df,
            feature_refs=["weather_stats:t2m_mean_24h"],
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 12, 31),
        )
    """

    def __init__(
        self,
        feast_repo_path: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        self._repo_path = feast_repo_path or FEAST_REPO_PATH
        self._redis = _connect_redis() if redis_url is None else self._init_redis(redis_url)
        self._store = self._init_feast()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_refs: List[str],
    ) -> pd.DataFrame:
        """
        Retrieve online features for a list of entity rows.
        Checks Redis cache (TTL 300s) before hitting the Feast online store.
        Returns a DataFrame with one row per entity.
        """
        entity_type = _entity_type_from_rows(entity_rows)
        cache_key = _redis_key(entity_rows, feature_refs)

        t0 = time.perf_counter()
        try:
            # --- Cache check ---
            cached = self._redis_get(cache_key)
            if cached is not None:
                _CACHE_HITS.labels(entity_type=entity_type).inc()
                return pd.DataFrame(cached)

            # --- Feast online store ---
            result = self._store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            ).to_df()

            # --- Store in Redis ---
            self._redis_set(cache_key, result.to_dict(orient="list"), ttl=ONLINE_CACHE_TTL)
            return result

        finally:
            _LATENCY.labels(method="get_online_features").observe(time.perf_counter() - t0)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve historical features for training/evaluation.
        No Redis cache — always reads from Feast offline store.
        entity_df must have 'event_timestamp' column and one entity key column.
        """
        t0 = time.perf_counter()
        try:
            # Filter entity_df by start/end if provided
            df = entity_df.copy()
            if start_dt is not None:
                df = df[df["event_timestamp"] >= pd.Timestamp(start_dt, tz="UTC")]
            if end_dt is not None:
                df = df[df["event_timestamp"] <= pd.Timestamp(end_dt, tz="UTC")]

            job = self._store.get_historical_features(
                entity_df=df,
                features=feature_refs,
            )
            return job.to_df()
        finally:
            _LATENCY.labels(method="get_historical_features").observe(time.perf_counter() - t0)

    @staticmethod
    def tag_mlflow_run(run: "mlflow.ActiveRun") -> None:
        """Tag an active MLflow run with feature_store_version."""
        mlflow.set_tag("feature_store_version", FEATURE_STORE_VERSION)
        logger.info("MLflow run tagged: feature_store_version=%s", FEATURE_STORE_VERSION)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_feast(self):
        try:
            from feast import FeatureStore  # type: ignore
            store = FeatureStore(repo_path=self._repo_path)
            logger.info("FeatureStoreClient: Feast store initialised from %s", self._repo_path)
            return store
        except ImportError:
            logger.warning("FeatureStoreClient: feast not installed — using stub store")
            return _FeastStub()
        except Exception as exc:
            logger.warning("FeatureStoreClient: Feast init failed (%s) — using stub store", exc)
            return _FeastStub()

    def _init_redis(self, redis_url: str):
        try:
            import redis  # type: ignore
            client = redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
            return client
        except Exception as exc:
            logger.warning("FeatureStoreClient: Redis init failed (%s)", exc)
            return None

    def _redis_get(self, key: str) -> Optional[dict]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("FeatureStoreClient: Redis GET failed (%s)", exc)
            return None

    def _redis_set(self, key: str, data: dict, ttl: int) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(key, ttl, json.dumps(data, default=str))
        except Exception as exc:
            logger.debug("FeatureStoreClient: Redis SET failed (%s)", exc)


# ---------------------------------------------------------------------------
# Stub (when Feast is not installed or repo path absent)
# ---------------------------------------------------------------------------

class _FeastStub:
    """No-op Feast stub — returns empty DataFrames and logs a warning."""

    def get_online_features(self, features, entity_rows):
        logger.warning("_FeastStub.get_online_features called — no real Feast store available")
        return _StubResult(pd.DataFrame())

    def get_historical_features(self, entity_df, features):
        logger.warning("_FeastStub.get_historical_features called — no real Feast store available")
        return _StubJob(pd.DataFrame())


class _StubResult:
    def __init__(self, df): self._df = df
    def to_df(self): return self._df


class _StubJob:
    def __init__(self, df): self._df = df
    def to_df(self): return self._df
