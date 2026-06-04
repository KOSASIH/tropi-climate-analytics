"""
Automated model retraining pipeline with A/B testing and champion/challenger management.
Runs on schedule: XGBoost weekly (Mon 01:00 WIB), Prophet monthly (1st 02:00 WIB),
CNN quarterly (1st Jan/Apr/Jul/Oct 03:00 WIB).
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .mlflow_setup import ModelTracker, register_model, promote_model, ModelRegistry
from .feature_engineering import FeatureMatrix
from .models import PrecipNowcastXGB, SeasonalProphet, SatelliteClassifierCNN

logger = logging.getLogger(__name__)


# ── Enums & constants ──────────────────────────────────────────────────────────

class ModelType(str, Enum):
    XGB_PRECIP    = "xgb_precip"
    PROPHET       = "prophet"
    CNN_LANDCOVER = "cnn_landcover"
    CNN_CLOUD     = "cnn_cloud"
    LSTM_STREAM   = "lstm_stream"


RETRAIN_SCHEDULES = {
    ModelType.XGB_PRECIP:    "0 18 * * 0",       # Sun 01:00 WIB (UTC+7)
    ModelType.PROPHET:       "0 19 1 * *",        # 1st of month 02:00 WIB
    ModelType.CNN_LANDCOVER: "0 20 1 1,4,7,10 *", # Quarterly 03:00 WIB
    ModelType.CNN_CLOUD:     "0 20 1 1,4,7,10 *",
    ModelType.LSTM_STREAM:   "0 19 1 * *",
}

PERFORMANCE_THRESHOLDS = {
    ModelType.XGB_PRECIP:    {"rmse_24h": 8.0,  "mae_24h": 5.0},   # mm
    ModelType.PROPHET:       {"mape_monthly": 0.15},                 # 15%
    ModelType.CNN_LANDCOVER: {"accuracy": 0.85, "f1_macro": 0.80},
    ModelType.CNN_CLOUD:     {"accuracy": 0.92, "f1_macro": 0.90},
    ModelType.LSTM_STREAM:   {"nse": 0.80,      "rmse": 50.0},      # m³/s
}


# ── Data loading stubs (filled by DATA-FLOW feature store) ────────────────────

def load_training_window(
    model_type: ModelType,
    end_date: Optional[datetime] = None,
    lookback_days: int = 365,
) -> Tuple[pd.DataFrame, Any]:
    """
    Load training data from the feature store for the given model type.
    Returns (features_df, targets). Actual implementation calls DATA-FLOW API.
    """
    end = end_date or datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    logger.info("Loading %s training data: %s → %s", model_type, start.date(), end.date())
    # TODO: integrate with DATA-FLOW Kafka feature store
    raise NotImplementedError("DATA-FLOW feature store integration required")


def load_validation_window(
    model_type: ModelType,
    end_date: Optional[datetime] = None,
    holdout_days: int = 90,
) -> Tuple[pd.DataFrame, Any]:
    end = end_date or datetime.utcnow()
    start = end - timedelta(days=holdout_days)
    logger.info("Loading %s validation data: %s → %s", model_type, start.date(), end.date())
    raise NotImplementedError("DATA-FLOW feature store integration required")


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: Any,
    model_type: ModelType,
) -> Dict[str, float]:
    """Dispatch to the appropriate evaluation function by model type."""
    if model_type == ModelType.XGB_PRECIP:
        return model.score(X_val, y_val)
    if model_type == ModelType.PROPHET:
        from sklearn.metrics import mean_absolute_percentage_error
        preds = model.predict(horizon_days=len(X_val))
        metrics = {}
        for var, fc in preds.items():
            merged = fc.merge(y_val[var], on="ds", how="inner")
            metrics[f"mape_{var}"] = float(
                mean_absolute_percentage_error(merged["y"], merged["yhat"])
            )
        return metrics
    if model_type in (ModelType.CNN_LANDCOVER, ModelType.CNN_CLOUD):
        from sklearn.metrics import accuracy_score, f1_score
        import torch
        logits = model.predict(torch.tensor(X_val, dtype=torch.float32))
        preds  = logits.numpy()
        return {
            "accuracy": float(accuracy_score(y_val, preds)),
            "f1_macro": float(f1_score(y_val, preds, average="macro")),
        }
    return {}


def meets_threshold(metrics: Dict[str, float], model_type: ModelType) -> bool:
    thresholds = PERFORMANCE_THRESHOLDS.get(model_type, {})
    for metric, threshold in thresholds.items():
        value = metrics.get(metric)
        if value is None:
            logger.warning("Metric '%s' missing from evaluation results", metric)
            return False
        # Lower is better for error metrics; higher for accuracy/NSE/F1
        if metric.startswith(("rmse", "mae", "mape")):
            if value > threshold:
                logger.info("Metric %s=%.4f exceeds threshold %.4f", metric, value, threshold)
                return False
        else:
            if value < threshold:
                logger.info("Metric %s=%.4f below threshold %.4f", metric, value, threshold)
                return False
    return True


# ── A/B testing ────────────────────────────────────────────────────────────────

class ABTestManager:
    """
    Manages champion/challenger routing for A/B model evaluation.
    Challenger receives `challenger_traffic_pct`% of inference requests.
    Tracks online metrics and promotes challenger if it consistently outperforms champion.
    """

    def __init__(
        self,
        model_type: ModelType,
        champion_name: str,
        challenger_name: str,
        challenger_traffic_pct: float = 10.0,
        min_samples: int = 1000,
    ):
        self.model_type             = model_type
        self.champion_name          = champion_name
        self.challenger_name        = challenger_name
        self.challenger_traffic_pct = challenger_traffic_pct
        self.min_samples            = min_samples
        self._champion_metrics: List[float]   = []
        self._challenger_metrics: List[float] = []

    def route(self, request_id: str) -> str:
        """Deterministic routing by hashing request_id. Returns 'champion' or 'challenger'."""
        h = int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100
        return "challenger" if h < self.challenger_traffic_pct else "champion"

    def record(self, model: str, metric_value: float) -> None:
        if model == "champion":
            self._champion_metrics.append(metric_value)
        else:
            self._challenger_metrics.append(metric_value)

    def should_promote(self) -> Tuple[bool, str]:
        """
        Returns (should_promote, reason).
        Uses Welch's t-test; promotes if challenger is significantly better (p<0.05).
        """
        if (len(self._challenger_metrics) < self.min_samples or
                len(self._champion_metrics) < self.min_samples):
            return False, f"Insufficient samples (challenger={len(self._challenger_metrics)}, min={self.min_samples})"

        from scipy import stats
        t_stat, p_value = stats.ttest_ind(
            self._champion_metrics, self._challenger_metrics, equal_var=False
        )
        champ_mean = np.mean(self._champion_metrics)
        chall_mean = np.mean(self._challenger_metrics)
        better = chall_mean < champ_mean  # lower error is better

        if better and p_value < 0.05:
            reason = (f"Challenger ({chall_mean:.4f}) significantly better than "
                      f"champion ({champ_mean:.4f}), p={p_value:.4f}")
            return True, reason

        return False, f"Challenger not significantly better (p={p_value:.4f})"


# ── Retraining orchestrator ────────────────────────────────────────────────────

class RetrainingPipeline:
    """
    Orchestrates full retrain → evaluate → register → (optionally promote) cycle.
    Called by Airflow DAG; results logged to MLflow.
    """

    def __init__(self, model_type: ModelType, dry_run: bool = False):
        self.model_type = model_type
        self.dry_run    = dry_run

    def run(self, reference_date: Optional[datetime] = None) -> Dict[str, Any]:
        ref = reference_date or datetime.utcnow()
        run_name = f"retrain_{self.model_type}_{ref.strftime('%Y%m%d_%H%M')}"

        logger.info("Starting retraining: %s | %s", self.model_type, run_name)

        experiment_map = {
            ModelType.XGB_PRECIP:    "precipitation_nowcasting",
            ModelType.PROPHET:       "seasonal_forecasting",
            ModelType.CNN_LANDCOVER: "satellite_classification",
            ModelType.CNN_CLOUD:     "satellite_classification",
            ModelType.LSTM_STREAM:   "streamflow_forecasting",
        }

        with ModelTracker(experiment_map[self.model_type], run_name) as tracker:
            # 1. Load data
            try:
                X_train, y_train = load_training_window(self.model_type, end_date=ref)
                X_val,   y_val   = load_validation_window(self.model_type, end_date=ref)
            except NotImplementedError:
                logger.error("DATA-FLOW not connected — skipping retrain")
                return {"status": "skipped", "reason": "data_not_available"}

            # 2. Train
            model = self._build_and_train(X_train, y_train)

            # 3. Evaluate
            metrics = evaluate_model(model, X_val, y_val, self.model_type)
            tracker.log_metrics(metrics)
            logger.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))

            # 4. Quality gate
            if not meets_threshold(metrics, self.model_type):
                logger.warning("Model did NOT pass quality gate — not registering")
                return {"status": "rejected", "metrics": metrics}

            if self.dry_run:
                logger.info("Dry run — skipping registration")
                return {"status": "dry_run", "metrics": metrics}

            # 5. Register as Staging
            model_name = self._registry_name()
            run_uri    = tracker.log_model(model, artifact_path="model")
            version    = register_model(run_uri, model_name,
                                        description=f"Auto-retrained {ref.date()}")
            tracker.log_params({"registered_version": version, "model_name": model_name})

            return {
                "status":  "registered",
                "version": version,
                "metrics": metrics,
                "run_id":  tracker.run_id,
            }

    def _build_and_train(self, X_train: pd.DataFrame, y_train: Any) -> Any:
        if self.model_type == ModelType.XGB_PRECIP:
            m = PrecipNowcastXGB()
            m.fit(X_train, y_train)
            return m
        if self.model_type == ModelType.PROPHET:
            m = SeasonalProphet()
            m.fit(X_train)
            return m
        if self.model_type in (ModelType.CNN_LANDCOVER, ModelType.CNN_CLOUD):
            task = "land_cover" if self.model_type == ModelType.CNN_LANDCOVER else "cloud_mask"
            m = SatelliteClassifierCNN(task=task)
            return m  # CNN training loop implemented in train_cnn.py
        raise NotImplementedError(f"No trainer for {self.model_type}")

    def _registry_name(self) -> str:
        return {
            ModelType.XGB_PRECIP:    ModelRegistry.PRECIP_NOWCAST_XGB,
            ModelType.PROPHET:       ModelRegistry.SEASONAL_PROPHET,
            ModelType.CNN_LANDCOVER: ModelRegistry.LAND_COVER_CNN,
            ModelType.CNN_CLOUD:     ModelRegistry.CLOUD_MASK_CNN,
            ModelType.LSTM_STREAM:   ModelRegistry.STREAMFLOW_LSTM,
        }[self.model_type]


# ── Entry point for Airflow DAG ────────────────────────────────────────────────

def run_scheduled_retrain(model_type_str: str, **kwargs) -> Dict[str, Any]:
    """Airflow PythonOperator entry point."""
    model_type = ModelType(model_type_str)
    pipeline   = RetrainingPipeline(model_type)
    result     = pipeline.run()
    logger.info("Retrain result: %s", result)
    return result
