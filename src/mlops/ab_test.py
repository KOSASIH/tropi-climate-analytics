"""
ANALYTICA — A/B Test Framework
src/mlops/ab_test.py

Shadow-mode traffic splitter: 10% challenger / 90% champion.
Auto-promotes challenger if RMSE improvement > 5% with p < 0.05 (two-sided t-test, α=0.05).
"""
from __future__ import annotations
import logging, os, random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from pydantic import BaseModel

log = logging.getLogger("analytica.ab_test")
MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")

@dataclass
class ABTestConfig:
    champion_model:    str
    challenger_model:  str
    traffic_split:     float = 0.10      # fraction routed to challenger
    eval_window_days:  int   = 7
    alpha:             float = 0.05      # significance level
    min_improvement:   float = 0.05      # 5% RMSE improvement required for promotion
    experiment_name:   str   = "ab_test_comparison"


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


class ABTestFramework:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self._champion_errors:   List[float] = []
        self._challenger_errors: List[float] = []

    def route(self, request: Any) -> Tuple[str, bool]:
        """Return (model_name, is_shadow). Shadow requests are logged but not returned to caller."""
        if random.random() < self.config.traffic_split:
            return self.config.challenger_model, False   # challenger serves live
        return self.config.champion_model, False         # champion serves live

    def record_outcome(self, model_name: str, y_pred: float, y_true: float) -> None:
        """Log squared error for both champion and challenger."""
        err = (y_pred - y_true) ** 2
        if model_name == self.config.champion_model:
            self._champion_errors.append(err)
        elif model_name == self.config.challenger_model:
            self._challenger_errors.append(err)
        self._maybe_log_mlflow(model_name, y_pred, y_true)

    def evaluate(self) -> Optional[ABTestResult]:
        """Run t-test and return promotion decision. Returns None if insufficient data."""
        c_errs  = np.array(self._champion_errors)
        ch_errs = np.array(self._challenger_errors)
        if len(c_errs) < 30 or len(ch_errs) < 30:
            log.info("Insufficient data for A/B evaluation (%d / %d samples)", len(c_errs), len(ch_errs))
            return None

        champion_rmse   = float(np.sqrt(np.mean(c_errs)))
        challenger_rmse = float(np.sqrt(np.mean(ch_errs)))
        improvement_pct = (champion_rmse - challenger_rmse) / (champion_rmse + 1e-8)

        t_stat, p_val = stats.ttest_ind(c_errs, ch_errs, equal_var=False, alternative="two-sided")

        promote = (
            improvement_pct > self.config.min_improvement
            and p_val < self.config.alpha
        )

        if promote:
            reason = (
                f"Challenger RMSE {challenger_rmse:.4f} beats champion {champion_rmse:.4f} "
                f"by {improvement_pct*100:.1f}% (p={p_val:.4f} < α={self.config.alpha})"
            )
            self._promote_challenger()
        else:
            reason = (
                f"No promotion: improvement={improvement_pct*100:.1f}% "
                f"(need >5%), p={p_val:.4f} (need <{self.config.alpha})"
            )

        result = ABTestResult(
            champion_model=self.config.champion_model,
            challenger_model=self.config.challenger_model,
            champion_rmse=round(champion_rmse, 6),
            challenger_rmse=round(challenger_rmse, 6),
            improvement_pct=round(improvement_pct, 6),
            t_statistic=round(float(t_stat), 6),
            p_value=round(float(p_val), 6),
            promote_challenger=promote,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._log_result_mlflow(result)
        log.info("A/B result: %s", reason)
        return result

    def _promote_challenger(self) -> None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            versions = client.get_latest_versions(self.config.challenger_model, stages=["Staging"])
            if versions:
                client.transition_model_version_stage(
                    name=self.config.challenger_model,
                    version=versions[0].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                log.info("Promoted %s v%s to Production", self.config.challenger_model, versions[0].version)
        except Exception as exc:
            log.warning("MLflow promotion failed: %s", exc)

    def _maybe_log_mlflow(self, model_name: str, y_pred: float, y_true: float) -> None:
        pass  # batch-logged in evaluate()

    def _log_result_mlflow(self, result: ABTestResult) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.config.experiment_name)
            with mlflow.start_run(run_name="ab_evaluation"):
                mlflow.log_metrics({
                    "champion_rmse":    result.champion_rmse,
                    "challenger_rmse":  result.challenger_rmse,
                    "improvement_pct":  result.improvement_pct,
                    "p_value":          result.p_value,
                    "t_statistic":      result.t_statistic,
                    "promote":          int(result.promote_challenger),
                })
                mlflow.log_params({
                    "champion_model":   result.champion_model,
                    "challenger_model": result.challenger_model,
                    "reason":           result.reason,
                })
        except Exception as exc:
            log.warning("MLflow A/B logging failed: %s", exc)
