"""
ANALYTICA Sprint 4 — Daily Model Serving Health DAG (hardened)
dags/model_serving_health_dag.py

Schedule: 0 6 * * *  (06:00 WIB daily)

Stages:
  1. ping_inference_api
  2. run_drift_monitor_all_models
  3. log_drift_reports_to_mlflow
  4. evaluate_ab_test_results
  5. branch_on_drift (BranchPythonOperator)
     ├─ trigger_retraining_dag  (drift_detected=True)
     └─ skip_retrain            (no drift)
  6. notify_climate_os          (HTTP hook / MLflow tag climate_os_notified=true)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

log = logging.getLogger("analytica.model_serving_health_dag")

INFERENCE_API_URL   = os.getenv("ANALYTICA_INFERENCE_API_URL", "http://analytica-serving:8080")
MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
CLIMATE_OS_WEBHOOK  = os.getenv("CLIMATE_OS_WEBHOOK_URL", "")
RETRAINING_DAG_ID   = "analytica_retraining_pipeline"

ALL_MODELS = ["xgboost_nowcast", "prophet_seasonal", "cnn_land_cover", "lstm_streamflow"]

DEFAULT_ARGS = {
    "owner":            "analytica",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Ping inference API
# ─────────────────────────────────────────────────────────────────────────────

def _ping_inference_api(**ctx: Any) -> Dict[str, Any]:
    import requests
    url = f"{INFERENCE_API_URL}/health"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        unhealthy = [m for m, ok in data.get("models", {}).items() if not ok]
        if unhealthy:
            log.warning("Unhealthy models: %s", unhealthy)
        else:
            log.info("All inference API models healthy ✓")
        ctx["ti"].xcom_push(key="api_health", value=data)
        return data
    except Exception as exc:
        log.error("Inference API health check FAILED: %s", exc)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Run drift monitor for all models
# ─────────────────────────────────────────────────────────────────────────────

def _run_drift_monitor_all_models(**ctx: Any) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/opt/analytica/src")
    from monitoring.drift_monitor import DriftMonitor  # type: ignore

    summary: Dict[str, Any] = {
        "models_with_drift":   [],
        "reports":             {},
        "any_drift":           False,
        "any_force_retrain":   False,
    }
    for model in ALL_MODELS:
        try:
            report = DriftMonitor(model_name=model).run()
            saved  = report.save()
            summary["reports"][model] = {
                "drift_detected":    report.drift_detected,
                "force_retrain":     report.force_retrain,
                "max_psi":           report.max_psi,
                "recommended_action": report.recommended_action,
                "report_path":       str(saved),
                "mlflow_run_id":     report.mlflow_run_id,
            }
            if report.drift_detected:
                summary["models_with_drift"].append(model)
                summary["any_drift"] = True
            if report.force_retrain:
                summary["any_force_retrain"] = True
            log.info("[%s] drift=%s max_psi=%.4f action=%s",
                     model, report.drift_detected, report.max_psi, report.recommended_action)
        except Exception as exc:
            log.error("[%s] drift monitor failed: %s", model, exc)
            summary["reports"][model] = {"error": str(exc)}

    ctx["ti"].xcom_push(key="drift_summary", value=summary)
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Log drift reports to MLflow (explicit stage)
# ─────────────────────────────────────────────────────────────────────────────

def _log_drift_reports_to_mlflow(**ctx: Any) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        summary = ctx["ti"].xcom_pull(task_ids="run_drift_monitor_all_models", key="drift_summary") or {}
        mlflow.set_experiment("drift_monitoring_daily")
        with mlflow.start_run(run_name=f"daily_drift_{datetime.now(timezone.utc).strftime('%Y%m%d')}"):
            mlflow.log_param("models_checked", ",".join(ALL_MODELS))
            mlflow.log_metric("models_with_drift", len(summary.get("models_with_drift", [])))
            mlflow.log_metric("any_force_retrain", int(summary.get("any_force_retrain", False)))
            for model, report in summary.get("reports", {}).items():
                if isinstance(report, dict) and "max_psi" in report:
                    mlflow.log_metric(f"{model}_max_psi", report["max_psi"])
            mlflow.log_dict(summary, "drift_summary.json")
        log.info("Drift reports logged to MLflow experiment 'drift_monitoring_daily'")
    except Exception as exc:
        log.warning("MLflow drift report logging failed: %s", exc)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Evaluate A/B test results
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_ab_test_results(**ctx: Any) -> Dict[str, Any]:
    try:
        import sys
        sys.path.insert(0, "/opt/analytica/src")
        from mlops.ab_test import ABTestConfig, ABTestFramework  # type: ignore
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client  = MlflowClient()
        results = {}
        pairs   = [
            ("xgboost_nowcast",   "xgboost_nowcast_challenger"),
            ("lstm_streamflow",   "lstm_streamflow_challenger"),
            ("prophet_seasonal",  "prophet_seasonal_challenger"),
        ]
        for champion, challenger in pairs:
            champion_versions   = client.get_latest_versions(champion,    stages=["Production"])
            challenger_versions = client.get_latest_versions(challenger,  stages=["Staging"])
            if not champion_versions or not challenger_versions:
                log.info("Skipping A/B for %s — no Staging challenger", champion)
                continue
            config  = ABTestConfig(champion_model=champion, challenger_model=challenger)
            fw      = ABTestFramework(config)
            # Load buffered prediction errors from MLflow 'ab_testing' experiment
            runs = client.search_runs(
                experiment_ids=[client.get_experiment_by_name("ab_testing").experiment_id
                                if client.get_experiment_by_name("ab_testing") else "0"],
                filter_string=f"params.champion_model = '{champion}'",
                max_results=1, order_by=["start_time DESC"],
            )
            result = fw.evaluate()
            if result:
                results[champion] = result.model_dump()
                log.info("A/B result for %s: promote=%s", champion, result.promote_challenger)

        ctx["ti"].xcom_push(key="ab_results", value=results)
        return results
    except Exception as exc:
        log.warning("A/B evaluation failed: %s", exc)
        ctx["ti"].xcom_push(key="ab_results", value={})
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Branch: drift? → retrain / skip
# ─────────────────────────────────────────────────────────────────────────────

def _branch_on_drift(**ctx: Any) -> str:
    summary = ctx["ti"].xcom_pull(task_ids="run_drift_monitor_all_models", key="drift_summary") or {}
    if summary.get("any_force_retrain", False):
        log.info("Drift force_retrain=True for models: %s → triggering retraining", summary.get("models_with_drift"))
        return "trigger_retraining_dag"
    log.info("No force_retrain — skipping retraining")
    return "skip_retrain"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Notify CLIMATE-OS
# ─────────────────────────────────────────────────────────────────────────────

def _notify_climate_os(**ctx: Any) -> None:
    summary    = ctx["ti"].xcom_pull(task_ids="run_drift_monitor_all_models", key="drift_summary") or {}
    ab_results = ctx["ti"].xcom_pull(task_ids="evaluate_ab_test_results",     key="ab_results")   or {}
    payload = {
        "agent":            "ANALYTICA",
        "dag_run_id":       ctx["run_id"],
        "execution_date":   str(ctx["ds"]),
        "drift_summary":    summary,
        "ab_results":       ab_results,
        "reported_at":      datetime.now(timezone.utc).isoformat(),
    }

    # Try HTTP webhook first
    if CLIMATE_OS_WEBHOOK:
        try:
            import requests
            resp = requests.post(CLIMATE_OS_WEBHOOK, json=payload, timeout=10)
            resp.raise_for_status()
            log.info("CLIMATE-OS webhook notified (HTTP %s)", resp.status_code)
            return
        except Exception as exc:
            log.warning("CLIMATE-OS webhook failed (%s) — falling back to MLflow tag", exc)

    # Fallback: log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("drift_monitoring_daily")
        with mlflow.start_run(run_name="climate_os_notification", nested=True):
            mlflow.set_tag("climate_os_notified", "true")
            mlflow.set_tag("notification_ts", payload["reported_at"])
            mlflow.log_dict(payload, "climate_os_payload.json")
        log.info("CLIMATE-OS notification logged to MLflow (climate_os_notified=true)")
    except Exception as exc:
        log.error("CLIMATE-OS notification (all paths) failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="model_serving_health_dag",
    description=(
        "ANALYTICA Sprint 4 daily serving health + drift + A/B evaluation. "
        "BranchPythonOperator routes to retraining DAG on PSI>0.2. "
        "Final stage notifies CLIMATE-OS via HTTP webhook or MLflow tag."
    ),
    schedule_interval="0 6 * * *",   # 06:00 WIB
    start_date=days_ago(1),
    default_args=DEFAULT_ARGS,
    catchup=False,
    max_active_runs=1,
    tags=["analytica", "serving", "drift", "ab-test", "sprint4"],
) as dag:

    t1_health = PythonOperator(
        task_id="ping_inference_api",
        python_callable=_ping_inference_api,
        doc_md="GET /health on inference API — asserts all 4 models loaded.",
    )

    t2_drift = PythonOperator(
        task_id="run_drift_monitor_all_models",
        python_callable=_run_drift_monitor_all_models,
        doc_md="PSI + KS-test drift detection for all 4 models. Saves DriftMonitorReport JSON.",
    )

    t3_log = PythonOperator(
        task_id="log_drift_reports_to_mlflow",
        python_callable=_log_drift_reports_to_mlflow,
        doc_md="Log daily drift summary and per-model PSI to MLflow experiment 'drift_monitoring_daily'.",
    )

    t4_ab = PythonOperator(
        task_id="evaluate_ab_test_results",
        python_callable=_evaluate_ab_test_results,
        doc_md="Run ABTestFramework t-test for each champion/challenger pair. Auto-promotes if criteria met.",
    )

    t5_branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_branch_on_drift,
        doc_md="Branch: force_retrain=True → trigger_retraining_dag | else → skip_retrain.",
    )

    t6a_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id=RETRAINING_DAG_ID,
        conf={"triggered_by": "model_serving_health_dag", "force_retrain": True},
        wait_for_completion=False,
        doc_md=f"Trigger `{RETRAINING_DAG_ID}` with force_retrain=True for drifted models.",
    )

    t6b_skip = EmptyOperator(
        task_id="skip_retrain",
        doc_md="No drift requiring retraining — no-op.",
    )

    t7_notify = PythonOperator(
        task_id="notify_climate_os",
        python_callable=_notify_climate_os,
        trigger_rule="none_failed_min_one_success",
        doc_md="Notify CLIMATE-OS via HTTP webhook or MLflow tag climate_os_notified=true.",
    )

    # Pipeline
    t1_health >> t2_drift >> t3_log >> t4_ab >> t5_branch
    t5_branch >> [t6a_retrain, t6b_skip]
    [t6a_retrain, t6b_skip] >> t7_notify
