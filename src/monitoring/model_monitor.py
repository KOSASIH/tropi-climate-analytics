"""
Model Monitor — ANALYTICA Sprint 7 L1
Module: src/monitoring/model_monitor.py

Class: ModelMonitor
  check_data_drift(model_id, reference_df, current_df) → DriftReport
  check_performance_degradation(model_id, window_days=7) → PerformanceReport
  run_full_check(model_id) → MonitorReport

Drift detection:
  PSI per feature — threshold 0.2 (warning) / 0.25 (critical)
  KS test p-value < 0.05 → flag feature as drifted
  Jensen-Shannon divergence for categorical features (LULC class distribution)

Performance degradation:
  Rolling 7-day MAE/MAPE/NSE/macro-F1 vs baseline from MLflow production run tags
  MAE > baseline × 1.15 → DEGRADED | MAE > baseline × 1.30 → CRITICAL

On CRITICAL: set Airflow Variable RETRAIN_{MODEL_ID}=true
Prometheus: MODEL_DRIFT_SCORE{model_id, feature, drift_type=psi|ks|js} Gauge
Output JSON: workspace/output/monitoring/drift_{model_id}_{YYYYMMDD}.json
             workspace/output/monitoring/performance_{model_id}_{YYYYMMDD}.json
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
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("workspace/output/monitoring")

# Drift thresholds
PSI_WARNING  = 0.2
PSI_CRITICAL = 0.25
KS_ALPHA     = 0.05
JS_THRESHOLD = 0.1   # JS divergence threshold for categorical drift

# Performance thresholds (relative to baseline)
PERF_DEGRADED  = 1.15
PERF_CRITICAL  = 1.30

# Model → primary metric mapping
MODEL_METRIC_MAP: Dict[str, str] = {
    "xgb_precipitation":  "val_mae",
    "xgb_precip_nowcast": "val_mae",
    "prophet_seasonal":   "val_mape",
    "cnn_landcover":      "val_macro_f1",
    "lstm_streamflow":    "val_nse",
}

# For NSE/F1 lower is BETTER only for MAE/MAPE; for NSE/F1 we invert
HIGHER_IS_BETTER = {"val_nse", "val_macro_f1", "val_kge"}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    feature:       str
    psi:           Optional[float]
    ks_pvalue:     Optional[float]
    js_divergence: Optional[float]
    status:        str   # "ok" | "warning" | "critical"
    drift_types:   List[str] = field(default_factory=list)  # ["psi", "ks", "js"]


@dataclass
class DriftReport:
    model_id:          str
    run_date:          str
    n_reference_rows:  int
    n_current_rows:    int
    feature_results:   List[FeatureDriftResult] = field(default_factory=list)
    overall_status:    str = "ok"   # "ok" | "warning" | "critical"
    drifted_features:  List[str] = field(default_factory=list)
    report_path:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id":         self.model_id,
            "run_date":         self.run_date,
            "n_reference_rows": self.n_reference_rows,
            "n_current_rows":   self.n_current_rows,
            "overall_status":   self.overall_status,
            "drifted_features": self.drifted_features,
            "feature_results":  [vars(f) for f in self.feature_results],
            "report_path":      self.report_path,
        }


@dataclass
class PerformanceReport:
    model_id:          str
    run_date:          str
    window_days:       int
    metric_name:       str
    baseline_value:    Optional[float]
    current_value:     Optional[float]
    relative_change:   Optional[float]
    status:            str = "ok"   # "ok" | "degraded" | "critical" | "no_baseline"
    report_path:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id":        self.model_id,
            "run_date":        self.run_date,
            "window_days":     self.window_days,
            "metric_name":     self.metric_name,
            "baseline_value":  self.baseline_value,
            "current_value":   self.current_value,
            "relative_change": self.relative_change,
            "status":          self.status,
            "report_path":     self.report_path,
        }


@dataclass
class MonitorReport:
    model_id:     str
    run_date:     str
    drift:        Optional[DriftReport]        = None
    performance:  Optional[PerformanceReport]  = None
    overall_status: str = "ok"   # "ok" | "warning" | "degraded" | "critical"
    retrain_triggered: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id":         self.model_id,
            "run_date":         self.run_date,
            "overall_status":   self.overall_status,
            "retrain_triggered": self.retrain_triggered,
            "drift":            self.drift.to_dict()       if self.drift       else None,
            "performance":      self.performance.to_dict() if self.performance else None,
        }


# ---------------------------------------------------------------------------
# PSI helpers
# ---------------------------------------------------------------------------

def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index (PSI). PSI = Σ (A-E) * ln(A/E)."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Use quantile-based bins from reference
    bin_edges = np.quantile(ref_clean, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(cur_clean, bins=bin_edges)[0].astype(float)

    ref_pct = (ref_counts + 1e-6) / ref_counts.sum()
    cur_pct = (cur_counts + 1e-6) / cur_counts.sum()

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0,1]) for categorical distributions."""
    p = (p_counts + 1e-9) / (p_counts.sum() + 1e-9)
    q = (q_counts + 1e-9) / (q_counts.sum() + 1e-9)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ModelMonitor:
    """
    Data drift and performance degradation monitor for all 4 ANALYTICA production models.
    Called daily by analytica_model_monitoring DAG (L2).
    """

    def check_data_drift(
        self,
        model_id:      str,
        reference_df:  pd.DataFrame,
        current_df:    pd.DataFrame,
        run_date:      Optional[date] = None,
    ) -> DriftReport:
        """
        Compute per-feature drift scores between reference and current distributions.

        Args:
            model_id:      Model identifier.
            reference_df:  Baseline feature DataFrame (last 90 days).
            current_df:    Current feature DataFrame (last 7 days).
            run_date:      Date of this check (default: today UTC).

        Returns:
            DriftReport with per-feature PSI, KS, JS results.
        """
        run_date = run_date or date.today()
        yyyymmdd = run_date.strftime("%Y%m%d")

        feature_results: List[FeatureDriftResult] = []
        drifted: List[str] = []
        worst_status = "ok"

        # Align on common columns, skip ID / timestamp columns
        skip_patterns = ("id", "timestamp", "entity_id", "valid_time", "ds", "date")
        common_cols = [
            c for c in reference_df.columns
            if c in current_df.columns and not any(p in c.lower() for p in skip_patterns)
        ]

        try:
            from src.data.metrics import MODEL_DRIFT_SCORE
            _metrics = True
        except ImportError:
            _metrics = False

        for col in common_cols:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue

            drift_types: List[str] = []
            col_psi: Optional[float] = None
            col_ks:  Optional[float] = None
            col_js:  Optional[float] = None

            # Categorical: JS divergence
            if reference_df[col].dtype in (object, "category") or reference_df[col].nunique() <= 20:
                all_cats = list(set(reference_df[col].dropna().unique()) | set(current_df[col].dropna().unique()))
                ref_c = np.array([np.sum(reference_df[col] == c) for c in all_cats], dtype=float)
                cur_c = np.array([np.sum(current_df[col]   == c) for c in all_cats], dtype=float)
                col_js = _js_divergence(ref_c, cur_c)
                if col_js > JS_THRESHOLD:
                    drift_types.append("js")
            else:
                # Continuous: PSI
                col_psi = _psi(ref_vals, cur_vals)
                if col_psi >= PSI_WARNING:
                    drift_types.append("psi")

                # KS test
                ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
                col_ks = float(ks_pval)
                if ks_pval < KS_ALPHA:
                    drift_types.append("ks")

            # Determine status
            if not drift_types:
                status = "ok"
            elif col_psi is not None and col_psi >= PSI_CRITICAL:
                status = "critical"
            else:
                status = "warning"

            if status in ("warning", "critical"):
                drifted.append(col)
                if status == "critical" and worst_status != "critical":
                    worst_status = "critical"
                elif status == "warning" and worst_status == "ok":
                    worst_status = "warning"

            # Emit Prometheus gauges
            if _metrics:
                if col_psi is not None:
                    MODEL_DRIFT_SCORE.labels(model_id=model_id, feature=col, drift_type="psi").set(col_psi)
                if col_ks is not None:
                    MODEL_DRIFT_SCORE.labels(model_id=model_id, feature=col, drift_type="ks").set(col_ks)
                if col_js is not None:
                    MODEL_DRIFT_SCORE.labels(model_id=model_id, feature=col, drift_type="js").set(col_js)

            feature_results.append(FeatureDriftResult(
                feature=col, psi=col_psi, ks_pvalue=col_ks, js_divergence=col_js,
                status=status, drift_types=drift_types,
            ))

        report = DriftReport(
            model_id=model_id,
            run_date=run_date.isoformat(),
            n_reference_rows=len(reference_df),
            n_current_rows=len(current_df),
            feature_results=feature_results,
            overall_status=worst_status,
            drifted_features=drifted,
        )

        report.report_path = self._write_json(
            OUTPUT_DIR / f"drift_{model_id}_{yyyymmdd}.json", report.to_dict()
        )
        logger.info(
            "Drift check [%s] status=%s drifted_features=%d/%d",
            model_id, worst_status, len(drifted), len(feature_results),
        )
        return report

    def check_performance_degradation(
        self,
        model_id:    str,
        window_days: int = 7,
        run_date:    Optional[date] = None,
    ) -> PerformanceReport:
        """
        Compare rolling window metric vs MLflow production baseline.

        Reads recent prediction errors from workspace/output/monitoring/predictions_{model_id}_*.json
        and compares against the baseline metric stored in the production MLflow run.
        """
        run_date = run_date or date.today()
        yyyymmdd = run_date.strftime("%Y%m%d")
        metric_name = MODEL_METRIC_MAP.get(model_id, "val_mae")

        baseline_value: Optional[float] = None
        current_value:  Optional[float] = None

        # Fetch baseline from MLflow production run
        try:
            import mlflow
            client = mlflow.MlflowClient()
            mvs = client.get_latest_versions(model_id, stages=["Production"])
            if mvs:
                run_data = client.get_run(mvs[0].run_id).data
                baseline_value = run_data.metrics.get(metric_name)
        except Exception as exc:
            logger.warning("Could not fetch MLflow baseline for %s: %s", model_id, exc)

        # Fetch current rolling metric from prediction logs
        pred_dir = OUTPUT_DIR
        pred_files = sorted(pred_dir.glob(f"predictions_{model_id}_*.json"))
        recent_errors: List[float] = []
        for pf in pred_files[-window_days:]:
            try:
                data = json.loads(pf.read_text())
                if isinstance(data.get("errors"), list):
                    recent_errors.extend(data["errors"])
                elif data.get("metric_value") is not None:
                    recent_errors.append(data["metric_value"])
            except Exception:
                pass

        if recent_errors:
            current_value = float(np.mean(recent_errors))

        # Determine status
        status = "ok"
        relative_change: Optional[float] = None

        if baseline_value is None or current_value is None:
            status = "no_baseline" if baseline_value is None else "ok"
        else:
            is_higher_better = metric_name in HIGHER_IS_BETTER
            if is_higher_better:
                relative_change = float((baseline_value - current_value) / (abs(baseline_value) + 1e-8))
                if relative_change > 0.30:
                    status = "critical"
                elif relative_change > 0.15:
                    status = "degraded"
            else:
                relative_change = float(current_value / (baseline_value + 1e-8))
                if relative_change > PERF_CRITICAL:
                    status = "critical"
                elif relative_change > PERF_DEGRADED:
                    status = "degraded"

        report = PerformanceReport(
            model_id=model_id,
            run_date=run_date.isoformat(),
            window_days=window_days,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            relative_change=relative_change,
            status=status,
        )
        report.report_path = self._write_json(
            OUTPUT_DIR / f"performance_{model_id}_{yyyymmdd}.json", report.to_dict()
        )
        logger.info(
            "Performance check [%s] metric=%s status=%s baseline=%.4f current=%.4f",
            model_id, metric_name, status,
            baseline_value or 0, current_value or 0,
        )
        return report

    def run_full_check(
        self,
        model_id:     str,
        reference_df: Optional[pd.DataFrame] = None,
        current_df:   Optional[pd.DataFrame] = None,
        run_date:     Optional[date] = None,
    ) -> MonitorReport:
        """
        Run drift + performance checks and trigger retraining if CRITICAL.

        Args:
            model_id:     Model identifier.
            reference_df: Passed through to check_data_drift (required for drift check).
            current_df:   Passed through to check_data_drift.
            run_date:     Date of this check.
        """
        run_date = run_date or date.today()
        drift_report: Optional[DriftReport]       = None
        perf_report:  Optional[PerformanceReport] = None

        if reference_df is not None and current_df is not None:
            drift_report = self.check_data_drift(model_id, reference_df, current_df, run_date)
        perf_report = self.check_performance_degradation(model_id, run_date=run_date)

        # Aggregate status
        statuses = [
            drift_report.overall_status if drift_report else "ok",
            perf_report.status          if perf_report  else "ok",
        ]
        if "critical" in statuses:
            overall = "critical"
        elif "degraded" in statuses:
            overall = "degraded"
        elif "warning" in statuses:
            overall = "warning"
        else:
            overall = "ok"

        # Trigger retraining on CRITICAL
        retrain_triggered = False
        if overall == "critical":
            retrain_triggered = self._trigger_retraining(model_id)

        report = MonitorReport(
            model_id=model_id,
            run_date=run_date.isoformat(),
            drift=drift_report,
            performance=perf_report,
            overall_status=overall,
            retrain_triggered=retrain_triggered,
        )
        logger.info(
            "Full check [%s] overall=%s retrain_triggered=%s",
            model_id, overall, retrain_triggered,
        )
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _trigger_retraining(model_id: str) -> bool:
        """Set Airflow Variable RETRAIN_{MODEL_ID}=true to gate Sprint 6 DAGs."""
        variable_map = {
            "xgb_precipitation":  "RETRAIN_XGB",
            "xgb_precip_nowcast": "RETRAIN_XGB",
            "prophet_seasonal":   "RETRAIN_PROPHET",
            "cnn_landcover":      "RETRAIN_CNN",
            "lstm_streamflow":    "RETRAIN_LSTM",
        }
        var_name = variable_map.get(model_id)
        if not var_name:
            # Fallback: uppercase MODEL_ID
            var_name = f"RETRAIN_{model_id.upper().replace('-', '_')}"
        try:
            from airflow.models import Variable
            Variable.set(var_name, "true")
            logger.warning("CRITICAL: set Airflow Variable %s=true for %s", var_name, model_id)
            return True
        except Exception as exc:
            logger.error("Failed to set retraining variable %s: %s", var_name, exc)
            return False

    @staticmethod
    def _write_json(path: Path, payload: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["written_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)
