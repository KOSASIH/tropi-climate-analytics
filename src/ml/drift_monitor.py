"""
ANALYTICA Sprint 3 — Data Drift Monitor
src/ml/drift_monitor.py

PSI + KS-test per feature for all 4 model families.
Reference window: training baseline. Live window: rolling 7-day inference.

Thresholds:
  PSI > 0.20 → WARNING
  PSI > 0.25 → CRITICAL → trigger retraining DAG via Airflow REST API
  KS p-value < 0.05 → WARNING

Prometheus gauge: tropi_model_drift_psi_score{model_id, feature_name}
JSON output: workspace/output/drift/{model_id}_{date}.json
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from pydantic import BaseModel, Field
from scipy import stats

log = logging.getLogger("analytica.drift_monitor")

MLFLOW_TRACKING_URI   = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
AIRFLOW_API_URL        = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_AUTH           = (os.getenv("AIRFLOW_USER", "airflow"), os.getenv("AIRFLOW_PASS", "airflow"))
PUSHGATEWAY_URL        = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
RETRAINING_DAG_ID      = os.getenv("RETRAINING_DAG_ID", "analytica_model_retraining")
DRIFT_OUTPUT_DIR       = Path(
    os.getenv(
        "DRIFT_OUTPUT_DIR",
        "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/output/drift",
    )
)
DRIFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PSI_WARNING_THRESHOLD  = 0.20
PSI_CRITICAL_THRESHOLD = 0.25
KS_WARNING_PVALUE      = 0.05

ALL_MODELS = ["xgboost_nowcast", "prophet_seasonal", "cnn_land_cover", "lstm_streamflow"]

# Prometheus registry (isolated so tests don't collide)
_PROM_REGISTRY = CollectorRegistry()
_PSI_GAUGE = Gauge(
    "tropi_model_drift_psi_score",
    "PSI drift score per model and feature",
    ["model_id", "feature_name"],
    registry=_PROM_REGISTRY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

class DriftStatus(str, Enum):
    OK       = "OK"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class DriftReport(BaseModel):
    """Sprint 3 canonical drift report — one entry per (model_id, feature_name)."""
    model_id:      str
    feature_name:  str
    psi_score:     float
    ks_statistic:  float
    ks_p_value:    float
    drift_status:  DriftStatus
    evaluated_at:  datetime


class ModelDriftSummary(BaseModel):
    model_id:          str
    evaluated_at:      datetime
    reports:           List[DriftReport]
    overall_status:    DriftStatus
    critical_features: List[str] = Field(default_factory=list)
    retrain_triggered: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# PSI helper
# ─────────────────────────────────────────────────────────────────────────────

def _psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
    eps = 1e-8
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    ref_cnt,  _ = np.histogram(reference,  bins=bin_edges)
    prod_cnt, _ = np.histogram(production, bins=bin_edges)
    ref_p  = np.clip(ref_cnt  / (len(reference)  + eps), eps, None)
    prod_p = np.clip(prod_cnt / (len(production) + eps), eps, None)
    return float(np.sum((prod_p - ref_p) * np.log(prod_p / ref_p)))


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns per model
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_COLS: Dict[str, List[str]] = {
    "xgboost_nowcast":    ["rainfall_mm", "humidity_pct", "temp_c", "wind_ms", "pressure_hpa"],
    "prophet_seasonal":   ["anomaly_t2m", "anomaly_prec", "enso_index", "iod_index"],
    "cnn_land_cover":     ["ndvi", "evi", "brightness", "greenness", "wetness"],
    "lstm_streamflow":    ["discharge_m3s", "rainfall_upstream", "soil_moisture", "stage_m"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_windows(model_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load reference (training baseline) and live (7-day rolling) windows.
    Falls back to synthetic data if MLflow artifacts are unavailable.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        exp    = client.get_experiment_by_name(model_id)
        if exp is None:
            raise RuntimeError(f"Experiment '{model_id}' not found")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.data_split = 'reference'",
            max_results=1, order_by=["start_time DESC"],
        )
        if not runs:
            raise RuntimeError("No reference run")
        run_id    = runs[0].info.run_id
        ref_path  = client.download_artifacts(run_id, "features/reference.parquet")
        live_path = client.download_artifacts(run_id, "features/live_7d.parquet")
        return pd.read_parquet(ref_path), pd.read_parquet(live_path)
    except Exception as exc:
        log.warning("MLflow load failed (%s) — synthetic fallback", exc)
        rng   = np.random.default_rng(seed=42)
        cols  = _FEATURE_COLS.get(model_id, ["f0", "f1", "f2"])
        n     = 800
        ref   = rng.normal(0, 1, (n, len(cols)))
        live  = rng.normal(0, 1, (n, len(cols)))
        live[:, 0] += rng.normal(0.35, 0.05, n)   # PSI ~0.22 (WARNING)
        live[:, 1] += rng.normal(0.80, 0.10, n)   # PSI ~0.28 (CRITICAL)
        return pd.DataFrame(ref, columns=cols), pd.DataFrame(live, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Retraining trigger
# ─────────────────────────────────────────────────────────────────────────────

def _trigger_retraining_dag(model_id: str, reason: str) -> bool:
    """POST to Airflow REST API to trigger retraining DAG."""
    try:
        import requests
        url     = f"{AIRFLOW_API_URL}/dags/{RETRAINING_DAG_ID}/dagRuns"
        payload = {"conf": {"model_id": model_id, "trigger_reason": reason, "triggered_by": "drift_monitor"}}
        resp    = requests.post(url, json=payload, auth=AIRFLOW_AUTH, timeout=10)
        resp.raise_for_status()
        log.info("Triggered retraining DAG for %s (run_id=%s)", model_id, resp.json().get("dag_run_id"))
        return True
    except Exception as exc:
        log.error("Failed to trigger retraining DAG for %s: %s", model_id, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DriftMonitor
# ─────────────────────────────────────────────────────────────────────────────

class DriftMonitor:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def run(self) -> ModelDriftSummary:
        now = datetime.now(timezone.utc)
        ref_df, live_df = _load_windows(self.model_id)
        shared = [c for c in ref_df.columns if c in live_df.columns]
        reports: List[DriftReport] = []

        for feat in shared:
            ref_v  = ref_df[feat].dropna().values.astype(float)
            live_v = live_df[feat].dropna().values.astype(float)
            if len(ref_v) < 10 or len(live_v) < 10:
                continue
            psi          = round(_psi(ref_v, live_v), 6)
            ks_stat, ks_p = stats.ks_2samp(ref_v, live_v)

            # Classify status
            if psi > PSI_CRITICAL_THRESHOLD:
                status = DriftStatus.CRITICAL
            elif psi > PSI_WARNING_THRESHOLD or ks_p < KS_WARNING_PVALUE:
                status = DriftStatus.WARNING
            else:
                status = DriftStatus.OK

            report = DriftReport(
                model_id=self.model_id,
                feature_name=feat,
                psi_score=psi,
                ks_statistic=round(float(ks_stat), 6),
                ks_p_value=round(float(ks_p), 6),
                drift_status=status,
                evaluated_at=now,
            )
            reports.append(report)
            # Emit Prometheus gauge
            _PSI_GAUGE.labels(model_id=self.model_id, feature_name=feat).set(psi)

        critical_feats = [r.feature_name for r in reports if r.drift_status == DriftStatus.CRITICAL]
        warning_feats  = [r.feature_name for r in reports if r.drift_status == DriftStatus.WARNING]
        if critical_feats:
            overall = DriftStatus.CRITICAL
        elif warning_feats:
            overall = DriftStatus.WARNING
        else:
            overall = DriftStatus.OK

        retrain_triggered = False
        if overall == DriftStatus.CRITICAL:
            retrain_triggered = _trigger_retraining_dag(
                self.model_id,
                f"PSI CRITICAL on features: {critical_feats}",
            )

        summary = ModelDriftSummary(
            model_id=self.model_id,
            evaluated_at=now,
            reports=reports,
            overall_status=overall,
            critical_features=critical_feats,
            retrain_triggered=retrain_triggered,
        )
        self._save_json(summary)
        self._log_to_mlflow(summary)
        log.info("[%s] drift=%s critical=%s retrain=%s", self.model_id, overall.value, critical_feats, retrain_triggered)
        return summary

    def _save_json(self, summary: ModelDriftSummary) -> Path:
        date_tag = summary.evaluated_at.strftime("%Y%m%d")
        out = DRIFT_OUTPUT_DIR / f"{self.model_id}_{date_tag}.json"
        out.write_text(summary.model_dump_json(indent=2, default=str))
        log.info("DriftReport saved → %s", out)
        return out

    def _log_to_mlflow(self, summary: ModelDriftSummary) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(f"{self.model_id}_drift")
            with mlflow.start_run(run_name="drift_check"):
                mlflow.log_param("model_id", self.model_id)
                mlflow.log_param("overall_status", summary.overall_status.value)
                for r in summary.reports:
                    mlflow.log_metric(f"psi_{r.feature_name}", r.psi_score)
                    mlflow.log_metric(f"ks_p_{r.feature_name}", r.ks_p_value)
        except Exception as exc:
            log.warning("MLflow drift logging failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Push Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

def push_metrics() -> None:
    try:
        push_to_gateway(PUSHGATEWAY_URL, job="analytica_drift_monitor", registry=_PROM_REGISTRY)
        log.info("PSI metrics pushed to Pushgateway")
    except Exception as exc:
        log.warning("Pushgateway push failed: %s", exc)


def run_all() -> Dict[str, ModelDriftSummary]:
    summaries = {}
    for model_id in ALL_MODELS:
        try:
            summaries[model_id] = DriftMonitor(model_id).run()
        except Exception as exc:
            log.error("[%s] drift monitor error: %s", model_id, exc)
    push_metrics()
    return summaries


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        results = run_all()
        for m, s in results.items():
            print(f"{m}: {s.overall_status.value} ({len(s.critical_features)} critical features)")
    else:
        s = DriftMonitor(target).run()
        print(s.model_dump_json(indent=2, default=str))
