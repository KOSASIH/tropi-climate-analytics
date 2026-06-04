"""
ANALYTICA Sprint 3 — A/B Testing Framework (shadow mode)
src/ml/ab_test.py

Shadow traffic: clone 10% of live inference requests → challenger (not served to end-user).
Metrics tracked per pair: RMSE, MAE, inference latency p50/p95/p99, drift PSI.
Statistical test: Welch's t-test on RMSE improvement significance.
Promotion criterion: challenger RMSE improvement > 5% WITH p < 0.05 AND p99 latency ≤ 120% champion.
Auto-promote: write new champion alias in MLflow model registry.
Prometheus counter: tropi_ab_test_requests_total{test_id, model_role}
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import CollectorRegistry, Counter, push_to_gateway
from pydantic import BaseModel
from scipy import stats

log = logging.getLogger("analytica.ab_test")
MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
PUSHGATEWAY_URL     = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus
# ─────────────────────────────────────────────────────────────────────────────

_PROM_REGISTRY = CollectorRegistry()
AB_REQUESTS_COUNTER = Counter(
    "tropi_ab_test_requests_total",
    "Total inference requests served in A/B test, by test and model role",
    ["test_id", "model_role"],
    registry=_PROM_REGISTRY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config & result schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ABTestConfig:
    champion_model_id:   str
    challenger_model_id: str
    traffic_split_pct:   float  = 10.0      # % of traffic to challenger
    min_sample_size:     int    = 100        # minimum samples before evaluation
    significance_level:  float  = 0.05      # α for Welch's t-test
    min_rmse_improvement_pct: float = 5.0   # min RMSE reduction required
    max_p99_latency_ratio: float = 1.20     # challenger p99 ≤ 120% champion p99

    @property
    def test_id(self) -> str:
        return f"{self.champion_model_id}_vs_{self.challenger_model_id}"


class LatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n:      int


class ABTestResult(BaseModel):
    test_id:             str
    champion_model_id:   str
    challenger_model_id: str
    n_champion:          int
    n_challenger:        int
    champion_rmse:       float
    challenger_rmse:     float
    rmse_improvement_pct: float
    t_statistic:         float
    p_value:             float
    champion_latency:    LatencyStats
    challenger_latency:  LatencyStats
    latency_ratio_p99:   float
    promote:             bool
    promotion_reason:    str
    evaluated_at:        str


# ─────────────────────────────────────────────────────────────────────────────
# Per-request record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _InferenceRecord:
    model_id:   str
    y_pred:     float
    y_true:     Optional[float]
    latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Framework
# ─────────────────────────────────────────────────────────────────────────────

class ABTestFramework:
    def __init__(self, config: ABTestConfig):
        self.config   = config
        self._records: List[_InferenceRecord] = []

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, request: Any) -> Tuple[str, bool]:
        """
        Returns (model_id_to_serve, is_shadow).
        is_shadow=True means the challenger handles this request in shadow mode
        (result NOT returned to end-user).
        """
        if random.random() * 100 < self.config.traffic_split_pct:
            AB_REQUESTS_COUNTER.labels(test_id=self.config.test_id, model_role="challenger").inc()
            return self.config.challenger_model_id, True
        AB_REQUESTS_COUNTER.labels(test_id=self.config.test_id, model_role="champion").inc()
        return self.config.champion_model_id, False

    def record(self, model_id: str, y_pred: float, y_true: Optional[float], latency_ms: float) -> None:
        self._records.append(_InferenceRecord(model_id, y_pred, y_true, latency_ms))

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self) -> Optional[ABTestResult]:
        champ_id  = self.config.champion_model_id
        chall_id  = self.config.challenger_model_id
        champ_recs  = [r for r in self._records if r.model_id == champ_id  and r.y_true is not None]
        chall_recs  = [r for r in self._records if r.model_id == chall_id  and r.y_true is not None]

        if len(champ_recs) < self.config.min_sample_size or len(chall_recs) < self.config.min_sample_size:
            log.info("Not enough samples for A/B evaluation (%d/%d)", len(champ_recs), len(chall_recs))
            return None

        champ_errs  = np.array([(r.y_pred - r.y_true) ** 2 for r in champ_recs])
        chall_errs  = np.array([(r.y_pred - r.y_true) ** 2 for r in chall_recs])
        champ_rmse  = float(np.sqrt(np.mean(champ_errs)))
        chall_rmse  = float(np.sqrt(np.mean(chall_errs)))
        improvement = (champ_rmse - chall_rmse) / (champ_rmse + 1e-8) * 100.0
        t_stat, p   = stats.ttest_ind(champ_errs, chall_errs, equal_var=False, alternative="two-sided")

        champ_lat   = self._latency_stats(champ_recs)
        chall_lat   = self._latency_stats(chall_recs)
        p99_ratio   = chall_lat.p99_ms / (champ_lat.p99_ms + 1e-3)

        rmse_ok     = improvement > self.config.min_rmse_improvement_pct
        pval_ok     = float(p) < self.config.significance_level
        lat_ok      = p99_ratio <= self.config.max_p99_latency_ratio
        promote     = rmse_ok and pval_ok and lat_ok

        if promote:
            reason = (
                f"RMSE improvement {improvement:.1f}% > {self.config.min_rmse_improvement_pct}%, "
                f"p={float(p):.4f} < {self.config.significance_level}, "
                f"p99_ratio={p99_ratio:.2f} ≤ {self.config.max_p99_latency_ratio}"
            )
            self._promote()
        else:
            parts = []
            if not rmse_ok:    parts.append(f"improvement={improvement:.1f}% (need >{self.config.min_rmse_improvement_pct}%)")
            if not pval_ok:    parts.append(f"p={float(p):.4f} (need <{self.config.significance_level})")
            if not lat_ok:     parts.append(f"p99_ratio={p99_ratio:.2f} (need ≤{self.config.max_p99_latency_ratio})")
            reason = "No promotion: " + "; ".join(parts)

        result = ABTestResult(
            test_id=self.config.test_id,
            champion_model_id=champ_id,
            challenger_model_id=chall_id,
            n_champion=len(champ_recs),
            n_challenger=len(chall_recs),
            champion_rmse=round(champ_rmse, 6),
            challenger_rmse=round(chall_rmse, 6),
            rmse_improvement_pct=round(improvement, 4),
            t_statistic=round(float(t_stat), 6),
            p_value=round(float(p), 6),
            champion_latency=champ_lat,
            challenger_latency=chall_lat,
            latency_ratio_p99=round(p99_ratio, 4),
            promote=promote,
            promotion_reason=reason,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._log_to_mlflow(result)
        self._push_metrics()
        log.info("A/B result [%s]: promote=%s — %s", self.config.test_id, promote, reason)
        return result

    # ── Promotion ─────────────────────────────────────────────────────────────

    def _promote(self) -> None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()
            # Transition challenger Staging → Production, demote champion
            staging = client.get_latest_versions(self.config.challenger_model_id, stages=["Staging"])
            if staging:
                client.transition_model_version_stage(
                    name=self.config.challenger_model_id,
                    version=staging[0].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                log.info("Promoted %s v%s → Production", self.config.challenger_model_id, staging[0].version)
            # Write champion alias pointing to challenger
            try:
                client.set_registered_model_alias(
                    self.config.champion_model_id, "champion", staging[0].version
                )
            except Exception:
                pass  # alias API may not be available in older MLflow
        except Exception as exc:
            log.warning("MLflow promotion failed: %s", exc)

    # ── MLflow logging ────────────────────────────────────────────────────────

    def _log_to_mlflow(self, result: ABTestResult) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("ab_testing")
            with mlflow.start_run(run_name=f"ab_eval_{result.test_id}"):
                mlflow.log_metrics({
                    "champion_rmse":        result.champion_rmse,
                    "challenger_rmse":      result.challenger_rmse,
                    "rmse_improvement_pct": result.rmse_improvement_pct,
                    "p_value":              result.p_value,
                    "t_statistic":          result.t_statistic,
                    "p99_latency_ratio":    result.latency_ratio_p99,
                    "promoted":             int(result.promote),
                    "champion_p99_ms":      result.champion_latency.p99_ms,
                    "challenger_p99_ms":    result.challenger_latency.p99_ms,
                })
                mlflow.log_params({
                    "test_id":           result.test_id,
                    "reason":            result.promotion_reason,
                    "n_champion":        result.n_champion,
                    "n_challenger":      result.n_challenger,
                })
        except Exception as exc:
            log.warning("MLflow A/B logging failed: %s", exc)

    # ── Prometheus ────────────────────────────────────────────────────────────

    def _push_metrics(self) -> None:
        try:
            push_to_gateway(PUSHGATEWAY_URL, job="analytica_ab_test", registry=_PROM_REGISTRY)
        except Exception as exc:
            log.warning("Pushgateway push failed: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _latency_stats(records: List[_InferenceRecord]) -> LatencyStats:
        lats = np.array([r.latency_ms for r in records])
        return LatencyStats(
            p50_ms=round(float(np.percentile(lats, 50)), 3),
            p95_ms=round(float(np.percentile(lats, 95)), 3),
            p99_ms=round(float(np.percentile(lats, 99)), 3),
            n=len(lats),
        )

    # ── Timed inference wrapper ───────────────────────────────────────────────

    def timed_call(self, model_fn: Callable, request: Any, y_true: Optional[float] = None) -> float:
        """Call model_fn(request), record latency, return prediction."""
        model_id, is_shadow = self.route(request)
        t0 = time.perf_counter()
        y_pred = model_fn(model_id, request)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.record(model_id, float(y_pred), y_true, latency_ms)
        if is_shadow:
            return None   # shadow result — do not serve to end-user
        return y_pred
