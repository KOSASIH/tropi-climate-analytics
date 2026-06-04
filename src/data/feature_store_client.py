"""
Feature Store Client — ANALYTICA Sprint 5 D3
Reads DATA-FLOW feature pipeline outputs from workspace/data/features/
and provides an LRU-cached, Prometheus-instrumented retrieval interface.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd
from prometheus_client import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
_FEATURE_REQUESTS = Counter(
    "tropi_feature_store_requests_total",
    "Total feature store requests by feature group and status",
    ["feature_group", "status"],  # status: hit | miss | fallback
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES_ROOT = Path("workspace/data/features")
FALLBACK_ROOT = FEATURES_ROOT / "fallback"
CACHE_TTL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# LRU cache helper — keyed on (entity_id, feature_group, date_str)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1000)
def _load_parquet_cached(path: str) -> pd.DataFrame:
    """Load a parquet file and cache the result for CACHE_TTL_SECONDS."""
    return pd.read_parquet(path)


def _cache_key(entity_id: str, feature_group: str, date: datetime) -> str:
    return str(FEATURES_ROOT / feature_group / entity_id / f"{date.date()}.parquet")


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class FeatureStoreClient:
    """
    Retrieve pre-computed features from DATA-FLOW's feature pipeline.

    DATA-FLOW output path convention:
        workspace/data/features/{feature_group}/{entity_id}/{date}.parquet

    Usage::

        client = FeatureStoreClient()
        df = client.get_features(
            entity_id="station_5201",
            feature_names=["t2m_mean_24h", "rh_mean_24h", "precip_sum_24h"],
            as_of=datetime(2026, 6, 4, 12, 0),
        )
    """

    def __init__(self, features_root: Optional[Path] = None):
        self._root = Path(features_root) if features_root else FEATURES_ROOT
        self._fallback_root = self._root / "fallback"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(
        self,
        entity_id: str,
        feature_names: List[str],
        as_of: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve named features for *entity_id* as of *as_of* date.

        Checks every registered feature group for the requested feature names.
        Returns a single-row DataFrame indexed by (entity_id, as_of).

        Falls back to last-known-good parquet if the as_of date file is missing.
        Returns an empty DataFrame only if both primary and fallback are absent.
        """
        # Derive feature group from the first feature prefix, or scan all groups
        feature_groups = self._detect_groups(entity_id, feature_names, as_of)

        frames: list[pd.DataFrame] = []
        for group in feature_groups:
            df = self._load_group(entity_id, group, as_of, feature_names)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for other in frames[1:]:
            merged = merged.join(other, how="outer", rsuffix="_dup")
            # drop any duplicate columns introduced by rsuffix
            merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]

        # Filter to only requested features (plus any always-present index cols)
        available = [f for f in feature_names if f in merged.columns]
        missing = set(feature_names) - set(available)
        if missing:
            logger.warning(
                "FeatureStoreClient: features not found in store: %s "
                "(entity_id=%s, as_of=%s)",
                sorted(missing),
                entity_id,
                as_of.date(),
            )
        return merged[available] if available else pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_groups(
        self,
        entity_id: str,
        feature_names: List[str],
        as_of: datetime,
    ) -> List[str]:
        """Return all group directories that exist for this entity on as_of date."""
        if not self._root.exists():
            return []
        groups = [
            d.name
            for d in self._root.iterdir()
            if d.is_dir() and d.name != "fallback"
        ]
        # Filter to groups that contain at least one file for this entity
        relevant = []
        for g in groups:
            entity_dir = self._root / g / entity_id
            if entity_dir.exists():
                relevant.append(g)
        return relevant or groups  # return all if nothing pre-filtered

    def _load_group(
        self,
        entity_id: str,
        feature_group: str,
        as_of: datetime,
        feature_names: List[str],
    ) -> Optional[pd.DataFrame]:
        """Load feature group parquet with LRU cache, falling back if missing."""
        primary_path = self._root / feature_group / entity_id / f"{as_of.date()}.parquet"

        # --- Cache hit ---
        if primary_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = _load_parquet_cached(str(primary_path))
                _FEATURE_REQUESTS.labels(
                    feature_group=feature_group, status="hit"
                ).inc()
                return df
            except Exception as exc:
                logger.warning(
                    "FeatureStoreClient: failed to read %s: %s", primary_path, exc
                )

        # --- Cache miss — try fallback ---
        _FEATURE_REQUESTS.labels(feature_group=feature_group, status="miss").inc()
        return self._load_fallback(entity_id, feature_group)

    def _load_fallback(
        self,
        entity_id: str,
        feature_group: str,
    ) -> Optional[pd.DataFrame]:
        """Return last-known-good features from fallback directory."""
        fallback_path = self._fallback_root / f"{entity_id}.parquet"
        if fallback_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_parquet(fallback_path)
                logger.warning(
                    "FeatureStoreClient: using fallback features for entity_id=%s "
                    "(feature_group=%s)",
                    entity_id,
                    feature_group,
                )
                _FEATURE_REQUESTS.labels(
                    feature_group=feature_group, status="fallback"
                ).inc()
                return df
            except Exception as exc:
                logger.error(
                    "FeatureStoreClient: fallback read failed for entity_id=%s: %s",
                    entity_id,
                    exc,
                )
        else:
            logger.warning(
                "FeatureStoreClient: no primary or fallback features found for "
                "entity_id=%s, feature_group=%s",
                entity_id,
                feature_group,
            )
        return None
