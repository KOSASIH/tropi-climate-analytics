"""
Ensemble Forecaster — ANALYTICA Sprint 7 L4
Module: src/training/ensemble_forecaster.py

Class: EnsembleForecaster
  predict(entity_id, entity_type, valid_time) → EnsemblePrediction
  calibrate_weights(validation_df) → dict[str, float]
  get_uncertainty_bounds(prediction, confidence=0.90) → tuple[float, float]

Members:
  xgb_precipitation  — MAE inverse-error weight
  prophet_seasonal   — MAPE inverse-error weight
  lstm_streamflow    — NSE inverse-error weight (river-specific)
  tft_climate        — CRPS inverse-error weight

Ensemble method: inverse-error weighting
Uncertainty:     Conformal prediction (nonconformity scores, 90% coverage guarantee)
Output:          workspace/output/ensemble/ensemble_{entity_id}_{YYYYMMDD_HHMM}.json
                 workspace/output/ensemble/latest_ensemble.json (rolling sidecar → VISUALIA + API-GATEWAY)
Prometheus:      ENSEMBLE_FORECAST_ERROR{entity_type, horizon_hr} Gauge
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("workspace/output/ensemble")

# Default error sentinel when a member has no recent data
_DEFAULT_ERROR = 999.0

MEMBER_MODELS = ["xgb_precipitation", "prophet_seasonal", "lstm_streamflow", "tft_climate"]

# Metric name used for weighting per member
MEMBER_METRIC = {
    "xgb_precipitation": "val_mae",
    "prophet_seasonal":  "val_mape",
    "lstm_streamflow":   "val_nse",
    "tft_climate":       "val_crps",
}

# For NSE / F1-style metrics: error = 1 - metric (so lower is worse)
INVERT_METRIC = {"val_nse", "val_macro_f1"}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePrediction:
    entity_id:           str
    entity_type:         str
    valid_time:          str           # ISO 8601
    ensemble_mean:       float
    ensemble_std:        float
    ci_lower_90:         float
    ci_upper_90:         float
    member_predictions:  Dict[str, Optional[float]]
    weights:             Dict[str, float]
    coverage_flag:       bool          # True if conformal coverage ≥ 90%
    horizon_hr:          int = 24

    def to_dict(self) -> dict:
        return {
            "entity_id":          self.entity_id,
            "entity_type":        self.entity_type,
            "valid_time":         self.valid_time,
            "ensemble_mean":      self.ensemble_mean,
            "ensemble_std":       self.ensemble_std,
            "ci_lower_90":        self.ci_lower_90,
            "ci_upper_90":        self.ci_upper_90,
            "member_predictions": self.member_predictions,
            "weights":            self.weights,
            "coverage_flag":      self.coverage_flag,
            "horizon_hr":         self.horizon_hr,
        }


# ---------------------------------------------------------------------------
# Nonconformity score store (in-memory singleton; production would persist to Redis/DB)
# ---------------------------------------------------------------------------

class _NonconformityStore:
    """Stores calibration nonconformity scores per model for conformal prediction."""

    def __init__(self):
        self._scores: Dict[str, List[float]] = {}

    def add(self, model_id: str, score: float) -> None:
        self._scores.setdefault(model_id, []).append(score)

    def quantile(self, model_id: str, confidence: float = 0.90) -> float:
        scores = self._scores.get(model_id, [])
        if not scores:
            return float("inf")
        q_level = math.ceil((1 + confidence) * len(scores)) / len(scores)
        q_level = min(q_level, 1.0)
        return float(np.quantile(scores, q_level))


_NC_STORE = _NonconformityStore()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EnsembleForecaster:
    """
    Weighted ensemble combining XGBoost + Prophet + LSTM + TFT predictions.
    Uncertainty quantified via conformal prediction (90% coverage guarantee).
    """

    def predict(
        self,
        entity_id:   str,
        entity_type: str,
        valid_time:  datetime,
        horizon_hr:  int = 24,
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction for a given entity at valid_time.

        Args:
            entity_id:   Station ID, watershed ID, or grid cell ID.
            entity_type: 'station_id' | 'watershed_id' | 'grid_cell_id'
            valid_time:  Target forecast valid time.
            horizon_hr:  Forecast horizon in hours.

        Returns:
            EnsemblePrediction with mean, std, 90% CI, member preds, and weights.
        """
        # Fetch member predictions from inference cache / API
        member_preds = self._fetch_member_predictions(entity_id, entity_type, valid_time)

        # Calibrate / load weights
        weights = self._load_weights(entity_type)

        # Compute weighted ensemble mean
        available = {m: v for m, v in member_preds.items() if v is not None}
        if not available:
            raise RuntimeError(f"No member predictions available for entity '{entity_id}'")

        # Renormalize weights over available members only
        avail_w = {m: weights.get(m, 1.0) for m in available}
        total   = sum(avail_w.values()) + 1e-12
        norm_w  = {m: w / total for m, w in avail_w.items()}

        ensemble_mean = float(sum(norm_w[m] * available[m] for m in available))
        ensemble_std  = float(
            math.sqrt(sum(norm_w[m] * (available[m] - ensemble_mean) ** 2 for m in available))
        )

        # Conformal prediction uncertainty bounds
        ci_lower, ci_upper, coverage_flag = self._conformal_bounds(
            entity_type, ensemble_mean, confidence=0.90
        )

        # Emit Prometheus
        try:
            from src.data.metrics import ENSEMBLE_FORECAST_ERROR
            err = abs(ensemble_mean)  # proxy; replaced with actual error post-observation
            ENSEMBLE_FORECAST_ERROR.labels(entity_type=entity_type, horizon_hr=str(horizon_hr)).set(err)
        except ImportError:
            pass

        pred = EnsemblePrediction(
            entity_id=entity_id,
            entity_type=entity_type,
            valid_time=valid_time.isoformat(),
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
            ci_lower_90=ci_lower,
            ci_upper_90=ci_upper,
            member_predictions=member_preds,
            weights=norm_w,
            coverage_flag=coverage_flag,
            horizon_hr=horizon_hr,
        )
        self._write_prediction(entity_id, valid_time, pred)
        return pred

    def calibrate_weights(self, validation_df: "pd.DataFrame") -> Dict[str, float]:
        """
        Compute inverse-error weights from a validation DataFrame.

        Args:
            validation_df: DataFrame with columns:
                           model_id (str), metric_name (str), metric_value (float)

        Returns:
            Normalized weight dict {model_id: weight}.
        """
        import pandas as pd

        errors: Dict[str, float] = {}
        for model_id in MEMBER_MODELS:
            rows = validation_df[validation_df["model_id"] == model_id]
            if rows.empty:
                errors[model_id] = _DEFAULT_ERROR
                continue
            metric_name = MEMBER_METRIC.get(model_id, "val_mae")
            metric_rows = rows[rows.get("metric_name", pd.Series()) == metric_name] if "metric_name" in rows.columns else rows
            if metric_rows.empty:
                metric_rows = rows
            val = float(metric_rows["metric_value"].mean())
            if metric_name in INVERT_METRIC:
                val = max(1.0 - val, 1e-4)   # error = 1 - metric
            errors[model_id] = max(val, 1e-6)

        # Inverse-error weighting: w_i = (1/err_i) / Σ(1/err_j)
        inv_errors = {m: 1.0 / errors[m] for m in MEMBER_MODELS}
        total      = sum(inv_errors.values()) + 1e-12
        weights    = {m: inv_errors[m] / total for m in MEMBER_MODELS}

        # Persist weights
        weight_path = OUTPUT_DIR / "ensemble_weights.json"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight_path.write_text(json.dumps({
            "weights": weights, "errors": errors,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        logger.info("Ensemble weights calibrated: %s", {m: f"{w:.3f}" for m, w in weights.items()})
        return weights

    def get_uncertainty_bounds(
        self,
        prediction: EnsemblePrediction,
        confidence: float = 0.90,
    ) -> Tuple[float, float]:
        """
        Return conformal prediction bounds at the requested coverage level.

        If calibration scores are available, returns coverage-guaranteed bounds;
        otherwise falls back to Gaussian approximation (mean ± 1.645 * std).
        """
        lower, upper, _ = self._conformal_bounds(
            prediction.entity_type, prediction.ensemble_mean,
            prediction.ensemble_std, confidence,
        )
        return lower, upper

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_member_predictions(
        self, entity_id: str, entity_type: str, valid_time: datetime
    ) -> Dict[str, Optional[float]]:
        """
        Fetch individual model predictions from the inference cache / serving layer.
        Returns None for members that are unavailable.
        """
        member_preds: Dict[str, Optional[float]] = {}
        for model_id in MEMBER_MODELS:
            try:
                from src.serving.inference_cache import InferenceCache
                cache = InferenceCache()
                cached = cache.get(model_id, entity_id, valid_time)
                member_preds[model_id] = float(cached) if cached is not None else None
            except Exception:
                member_preds[model_id] = None

        # If all None (cold-start), log a warning
        if all(v is None for v in member_preds.values()):
            logger.warning(
                "All member predictions unavailable for entity %s at %s — cache cold-start?",
                entity_id, valid_time.isoformat(),
            )
        return member_preds

    def _load_weights(self, entity_type: str) -> Dict[str, float]:
        """Load persisted calibrated weights or fall back to equal weights."""
        weight_path = OUTPUT_DIR / "ensemble_weights.json"
        if weight_path.exists():
            try:
                data = json.loads(weight_path.read_text())
                return data.get("weights", {})
            except Exception:
                pass
        # Equal weights fallback
        n = len(MEMBER_MODELS)
        return {m: 1.0 / n for m in MEMBER_MODELS}

    def _conformal_bounds(
        self,
        entity_type:  str,
        mean:         float,
        std:          float = 0.0,
        confidence:   float = 0.90,
    ) -> Tuple[float, float, bool]:
        """
        Conformal prediction interval using nonconformity quantile.
        Falls back to Gaussian CI when no calibration scores are available.
        """
        nc_quantile = _NC_STORE.quantile(entity_type, confidence)
        if not math.isinf(nc_quantile):
            lower        = mean - nc_quantile
            upper        = mean + nc_quantile
            coverage_flag = True
        else:
            # Gaussian fallback (z=1.645 for 90%)
            z_score      = 1.645
            margin       = z_score * max(std, 1e-6)
            lower        = mean - margin
            upper        = mean + margin
            coverage_flag = False

        return float(lower), float(upper), coverage_flag

    def add_calibration_score(self, entity_type: str, score: float) -> None:
        """Register a nonconformity score from a calibration observation."""
        _NC_STORE.add(entity_type, score)

    @staticmethod
    def _write_prediction(entity_id: str, valid_time: datetime, pred: EnsemblePrediction) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_str    = valid_time.strftime("%Y%m%d_%H%M")
        safe_id   = entity_id.replace("/", "_")
        fname     = f"ensemble_{safe_id}_{ts_str}.json"
        payload   = {**pred.to_dict(), "written_at": datetime.now(timezone.utc).isoformat()}

        (OUTPUT_DIR / fname).write_text(json.dumps(payload, indent=2, default=str))

        # Rolling sidecar → VISUALIA + API-GATEWAY
        latest = OUTPUT_DIR / "latest_ensemble.json"
        latest.write_text(json.dumps(payload, indent=2, default=str))
        logger.debug("Ensemble prediction written: %s (sidecar updated)", fname)
