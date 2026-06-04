"""
ANALYTICA Sprint 4 — A/B Test Framework (hardened)
src/mlops/ab_test.py

Shadow-mode: 10% challenger / 90% champion.
MLflow comparison experiment: 'ab_testing'.
Two-sided t-test (α=0.05) on 7-day rolling RMSE.
Auto-promote challenger → Production if RMSE improvement > 5% AND p < 0.05.
Emits tropi_ab_test_promotion_total counter on each auto-promote.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Counter
from pydantic import BaseModel
from scipy import stats

log = logging.getLogger("analytica.ab_test")
MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus counter
# ─────────────────────────────────────────────────────────────────────────────

AB_PROMOTION_COUNTER = Counter(
    "tropi_ab_test_promotion_total",
    "Total number of challenger→Production auto-promotions by A/B test framework",
    ["champion_model", "challenger_model"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Config & schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ABTestConfig:
    champion_model:     str
    challenger_model:   str
    traffic_split:      float = 0.10   # fraction to challenger
    eval_window_days:   int   = 7
    alpha:              float = 0.05
    min_improvement_pct: float = 5.0   # % RMSE improvement required for promotion
    experiment_name:    str   = "ab_testing"


class ABTestResult(BaseModel):
    champion_model:     str
    challenger_model:   str
    champion_rmse:      float
    challenger_rmse:    float
    improvement_pct:    float
    t_statistic:        float
    p_value:            float
    promote_challenger: bool
    reason:             str
    timestamp:          str


# ─────────────────────────────────────────────────────────────────────────────
# Framework
# ─────────────────────────────────────────────────────────────────────────────

class ABTestFramework:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self._champion_errors:   List[float] = []
        self._challenger_errors: List[float] = []
        # Per-prediction log for MLflow (y_pred, y_true, model)
        self._log_buffer: List[Dict[str, Any]] = []

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, request: Any) -> str:
        """Return model_name to serve this request (10% → challenger)."""
        if random.random() < self.config.traffic_split:
            return self.config.challenger_model
        return self.config.champion_model

    # ── Outcome recording ─────────────────────────────────────────────────────

    def record_outcome(self, model_name: str, y_pred: float, y_true: float) -> None:
        """Log squared error and buffer for MLflow."""
        err = (y_pred - y_true) ** 2
        if model_name == self.config.champion_model:
            self._champion_errors.append(err)
        elif model_name == self.config.challenger_model:
            self._challenger_errors.append(err)
        self._log_buffer.append({
            "model":    model_name,
            "y_pred":   y_pred,
            "y_true":   y_true,
            "sq_error": err,
        })

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self) -> Optional[ABTestResult]:
        """
        Two-sided t-test on squared errors.
        Auto-promotes if improvement > min_improvement_pct% AND p < alpha.
        Returns None if insufficient data (< 30 samples each).
        """
        c_errs  = np.array(self._champion_errors)
        ch_errs = np.array(self._challenger_errors)
        if len(c_errs) < 30 or len(ch_errs) < 30:
            log.info("Insufficient A/B data (%d champion / %d challenger)", len(c_errs), len(ch_errs))
            return None

        champion_rmse   = float(np.sqrt(np.mean(c_errs)))
        challenger_rmse = float(np.sqrt(np.mean(ch_errs)))
        improvement_pct = (champion_rmse - challenger_rmse) / (champion_rmse + 1e-8) * 100.0
        t_stat, p_val   = stats.ttest_ind(c_errs, ch_errs, equal_var=False, alternative="two-sided")

        min_pct  = self.config.min_improvement_pct
        promote  = improvement_pct > min_pct and p_val < self.config.alpha

        if promote:
            reason = (
                f"Challenger RMSE {challenger_rmse:.4f} beats champion {champion_rmse:.4f} "
                f"by {improvement_pct:.1f}% (p={p_val:.4f} < α={self.config.alpha}) → promoting to Production"
            )
            self._promote_challenger()
        else:
            reason = (
                f"No promotion: improvement={improvement_pct:.1f}% "
                f"(need >{min_pct}%), p={p_val:.4f} (need <{self.config.alpha})"
            )

        result = ABTestResult(
            champion_model=self.config.champion_model,
            challenger_model=self.config.challenger_model,
            champion_rmse=round(champion_rmse, 6),
            challenger_rmse=round(challenger_rmse, 6),
            improvement_pct=round(improvement_pct, 4),
            t_statistic=round(float(t_stat), 6),
            p_value=round(float(p_val), 6),
            promote_challenger=promote,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._log_to_mlflow(result)
        log.info("A/B evaluation: %s", reason)
        return result

    # ── Promotion ─────────────────────────────────────────────────────────────

    def _promote_challenger(self) -> None:
        """Transition challenger to Production in MLflow registry, emit Prometheus counter."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            staging_versions = client.get_latest_versions(
                self.config.challenger_model, stages=["Staging"]
            )
            if staging_versions:
                client.transition_model_version_stage(
                    name=self.config.challenger_model,
                    version=staging_versions[0].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                log.info(
                    "Promoted %s v%s → Production",
                    self.config.challenger_model,
                    staging_versions[0].version,
                )
            else:
                log.warning(
                    "No Staging version for %s — skipping MLflow transition",
                    self.config.challenger_model,
                )
        except Exception as exc:
            log.warning("MLflow promotion failed: %s", exc)

        # Always emit counter (even if MLflow step failed)
        AB_PROMOTION_COUNTER.labels(
            champion_model=self.config.champion_model,
            challenger_model=self.config.challenger_model,
        ).inc()
        log.info("tropi_ab_test_promotion_total incremented")

    # ── MLflow logging ────────────────────────────────────────────────────────

    def _log_to_mlflow(self, result: ABTestResult) -> None:
        try:
            import mlflow, pandas as pd
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.config.experiment_name)
            with mlflow.start_run(run_name="ab_evaluation"):
                mlflow.log_metrics({
                    "champion_rmse":    result.champion_rmse,
                    "challenger_rmse":  result.challenger_rmse,
                    "improvement_pct":  result.improvement_pct,
                    "p_value":          result.p_value,
                    "t_statistic":      result.t_statistic,
                    "promoted":         int(result.promote_challenger),
                })
                mlflow.log_params({
                    "champion_model":   result.champion_model,
                    "challenger_model": result.challenger_model,
                    "alpha":            self.config.alpha,
                    "min_improvement_pct": self.config.min_improvement_pct,
                    "reason":           result.reason,
                })
                # Log prediction buffer as CSV artifact
                if self._log_buffer:
                    import tempfile
                    df  = pd.DataFrame(self._log_buffer)
                    tmp = tempfile.mktemp(suffix=".csv")
                    df.to_csv(tmp, index=False)
                    mlflow.log_artifact(tmp, artifact_path="predictions")
        except Exception as exc:
            log.warning("MLflow A/B logging failed: %s", exc)
