"""
ANALYTICA — Model Serving Health DAG
dags/model_serving_health_dag.py

Daily DAG (0 6 * * * — 06:00 WIB) that:
  1. Pings /health on the ANALYTICA inference API
  2. Runs drift_monitor on all 4 models
  3. Logs DriftMonitorReport to MLflow
  4. Triggers retraining DAG for any model where drift_detected=True

Schedule: 0 6 * * *  (06:00 WIB / UTC+7 = 23:00 UTC previous day)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

log = logging.getLogger("analytica.model_serving_health_dag")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_API_URL = os.getenv("ANALYTICA_INFERENCE_API_URL", "http://analytica-serving:8080")
MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
RETRAINING_DAG_ID = "analytica_retraining_pipeline"

ALL_MODELS = [
    "xgboost_nowcast",
    "prophet_seasonal",
    "cnn_land_cover",
    "lstm_streamflow",
]

DEFAULT_ARGS = {
    "owner": "analytica",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────────────────────────────────────

def ping_inference_api(**context: Any) -> Dict[str, Any]:
    """Ping /health on the inference API and assert all models are loaded."""
    import requests
    url = f"{INFERENCE_API_URL}/health"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        health_data = resp.json()
        log.info("Inference API health: %s", health_data)
        loaded_models = health_data.get("models", {})
        unhealthy = [m for m, ok in loaded_models.items() if not ok]
        if unhealthy:
            log.warning("Models not loaded: %s", unhealthy)
            context["ti"].xcom_push(key="unhealthy_models", value=unhealthy)
        else:
            log.info("All models healthy ✓")
        context["ti"].xcom_push(key="api_healthy", value=True)
        return health_data
    except Exception as exc:
        log.error("Inference API health check FAILED: %s", exc)
        context["ti"].xcom_push(key="api_healthy", value=False)
        raise


def run_drift_monitor_all_models(**context: Any) -> Dict[str, Any]:
    """
    Run DriftMonitor for each model. Push per-model reports to XCom.
    Returns summary dict with drift_detected flag for any model.
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")
    from monitoring.drift_monitor import DriftMonitor  # type: ignore

    summary: Dict[str, Any] = {
        "models_with_drift": [],
        "reports": {},
        "any_drift": False,
        "any_force_retrain": False,
    }

    for model_name in ALL_MODELS:
        try:
            monitor = DriftMonitor(model_name=model_name)
            report  = monitor.run()
            saved   = report.save()

            summary["reports"][model_name] = {
                "drift_detected":    report.drift_detected,
                "force_retrain":     report.force_retrain,
                "max_psi":           report.max_psi,
                "recommended_action": report.recommended_action,
                "report_path":       str(saved),
                "mlflow_run_id":     report.mlflow_run_id,
            }

            if report.drift_detected:
                summary["models_with_drift"].append(model_name)
                summary["any_drift"] = True
            if report.force_retrain:
                summary["any_force_retrain"] = True

            log.info(
                "[%s] drift=%s max_psi=%.4f action=%s",
                model_name, report.drift_detected, report.max_psi, report.recommended_action,
            )
        except Exception as exc:
            log.error("[%s] Drift monitor failed: %s", model_name, exc)
            summary["reports"][model_name] = {"error": str(exc)}

    context["ti"].xcom_push(key="drift_summary", value=summary)
    return summary


def check_any_drift(**context: Any) -> bool:
    """ShortCircuit gate — returns True (proceed) only if drift detected."""
    summary = context["ti"].xcom_pull(task_ids="run_drift_monitor", key="drift_summary")
    if not summary:
        log.info("No drift summary found — skipping retraining trigger")
        return False
    any_drift = summary.get("any_force_retrain", False)
    if any_drift:
        log.info("Drift detected in models: %s — proceeding to retraining trigger",
                 summary.get("models_with_drift"))
    else:
        log.info("No force_retrain flags — ShortCircuit will stop DAG here")
    return any_drift


def build_retrain_conf(**context: Any) -> Dict[str, Any]:
    """Assemble retraining DAG config from drift summary."""
    summary = context["ti"].xcom_pull(task_ids="run_drift_monitor", key="drift_summary")
    models_to_retrain = summary.get("models_with_drift", ALL_MODELS)
    conf = {
        "triggered_by":        "model_serving_health_dag",
        "trigger_reason":      "drift_detected",
        "models_to_retrain":   models_to_retrain,
        "force_retrain":       True,
        "drift_reports":       summary.get("reports", {}),
        "trigger_timestamp":   datetime.utcnow().isoformat(),
    }
    log.info("Retraining conf: %s", json.dumps(conf, indent=2))
    context["ti"].xcom_push(key="retrain_conf", value=conf)
    return conf


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="model_serving_health_dag",
    description=(
        "Daily serving health check + drift monitor for all 4 ANALYTICA models. "
        "Auto-triggers retraining DAG when drift_detected=True (PSI > 0.2)."
    ),
    schedule_interval="0 6 * * *",   # 06:00 WIB daily  (CLOUD-FORGE spec)
    start_date=days_ago(1),
    default_args=DEFAULT_ARGS,
    catchup=False,
    max_active_runs=1,
    tags=["analytica", "serving", "drift", "mlops", "sprint3"],
) as dag:

    # ── T1: Ping inference API ───────────────────────────────────────────────
    t_api_health = PythonOperator(
        task_id="ping_inference_api",
        python_callable=ping_inference_api,
        doc_md=(
            "Ping GET /health on the ANALYTICA inference API. "
            "Asserts all 4 models are loaded. Raises on HTTP error."
        ),
    )

    # ── T2: Run drift monitor for all models ─────────────────────────────────
    t_drift_monitor = PythonOperator(
        task_id="run_drift_monitor",
        python_callable=run_drift_monitor_all_models,
        doc_md=(
            "Runs PSI + KS-test drift detection on each model's 30-day input distribution. "
            "Logs DriftMonitorReport to MLflow and workspace/output/drift/. "
            "Pushes drift_summary XCom for downstream gate."
        ),
    )

    # ── T3: ShortCircuit — only proceed if drift requires retraining ──────────
    t_drift_gate = ShortCircuitOperator(
        task_id="check_drift_requires_retrain",
        python_callable=check_any_drift,
        doc_md=(
            "ShortCircuit: proceeds downstream only when force_retrain=True "
            "(PSI > 0.2) for at least one model. Skips retraining trigger otherwise."
        ),
    )

    # ── T4: Build retraining config ───────────────────────────────────────────
    t_build_conf = PythonOperator(
        task_id="build_retrain_conf",
        python_callable=build_retrain_conf,
        doc_md="Assembles retraining DAG conf dict from drift_summary XCom.",
    )

    # ── T5: Trigger retraining DAG ────────────────────────────────────────────
    t_trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id=RETRAINING_DAG_ID,
        conf="{{ ti.xcom_pull(task_ids='build_retrain_conf', key='retrain_conf') }}",
        wait_for_completion=False,
        doc_md=(
            f"Triggers `{RETRAINING_DAG_ID}` with force_retrain=True and the list "
            "of drifted model names. Does not wait for completion."
        ),
    )

    # ── Task dependencies ─────────────────────────────────────────────────────
    t_api_health >> t_drift_monitor >> t_drift_gate >> t_build_conf >> t_trigger_retrain
