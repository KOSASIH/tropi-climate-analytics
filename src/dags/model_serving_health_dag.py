"""
ANALYTICA Sprint 3 — Model Serving Health Check DAG
src/dags/model_serving_health_dag.py

dag_id  : model_serving_health_check
Schedule: 0 6 * * *  (06:00 WIB / Asia/Jakarta)
SLA     : 10 minutes

Tasks:
  1. ping_health_endpoint       — GET /health → fail DAG if != 200
  2. check_model_staleness      — verify all 4 model labels saw inference in last 24 h
  3. run_drift_monitor          — run drift_monitor.py for all 4 models, log DriftReports
  4. check_challenger_and_ab    — scan MLflow registry for pending challenger → trigger ab_test.py
  5. push_summary_to_pushgateway — push aggregate metrics to Prometheus Pushgateway

Failure: emit tropi_model_serving_health_dag_failure metric → Prometheus alerting rule
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger("analytica.model_serving_health_check")

INFERENCE_API_URL   = os.getenv("ANALYTICA_INFERENCE_API_URL", "http://analytica-serving:8080")
MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
PUSHGATEWAY_URL     = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
PROMETHEUS_URL      = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

ALL_MODELS = ["xgboost_nowcast", "prophet_seasonal", "cnn_land_cover", "lstm_streamflow"]
INFERENCE_STALE_HOURS = 24

DEFAULT_ARGS = {
    "owner":            "analytica",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "email_on_failure": False,
    "sla":              timedelta(minutes=10),
}


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Ping /health
# ─────────────────────────────────────────────────────────────────────────────

def _ping_health_endpoint(**ctx: Any) -> Dict[str, Any]:
    import requests
    url = f"{INFERENCE_API_URL}/health"
    try:
        resp = requests.get(url, timeout=15)
    except Exception as exc:
        _push_failure_metric("ping_inference_api", str(exc))
        raise RuntimeError(f"Cannot reach inference API {url}: {exc}") from exc
    if resp.status_code != 200:
        _push_failure_metric("ping_inference_api", f"HTTP {resp.status_code}")
        raise RuntimeError(f"Inference API /health returned HTTP {resp.status_code}")
    data = resp.json()
    log.info("Inference API healthy: %s", data)
    ctx["ti"].xcom_push(key="health_payload", value=data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Check model staleness via Prometheus /metrics
# ─────────────────────────────────────────────────────────────────────────────

def _check_model_staleness(**ctx: Any) -> Dict[str, Any]:
    import requests
    stale_models = []
    cutoff_ts    = datetime.now(timezone.utc).timestamp() - INFERENCE_STALE_HOURS * 3600

    for model_id in ALL_MODELS:
        try:
            # Query Prometheus for last inference timestamp
            query = f'tropi_model_last_inference_timestamp{{model_id="{model_id}"}}'
            resp  = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query},
                timeout=10,
            )
            result = resp.json().get("data", {}).get("result", [])
            if not result:
                log.warning("[%s] No inference metrics found — marking stale", model_id)
                stale_models.append(model_id)
            else:
                last_ts = float(result[0]["value"][1])
                if last_ts < cutoff_ts:
                    log.warning("[%s] Last inference %.1f h ago — STALE", model_id,
                                (datetime.now(timezone.utc).timestamp() - last_ts) / 3600)
                    stale_models.append(model_id)
                else:
                    log.info("[%s] Inference seen within last %d h ✓", model_id, INFERENCE_STALE_HOURS)
        except Exception as exc:
            log.warning("[%s] Staleness check failed: %s — marking stale", model_id, exc)
            stale_models.append(model_id)

    if stale_models:
        log.warning("STALE models: %s", stale_models)
        _push_failure_metric("stale_models", f"stale={stale_models}")
    ctx["ti"].xcom_push(key="stale_models", value=stale_models)
    return {"stale_models": stale_models}


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Run drift monitor
# ─────────────────────────────────────────────────────────────────────────────

def _run_drift_monitor(**ctx: Any) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/opt/analytica/src")
    from ml.drift_monitor import run_all  # type: ignore
    summaries = run_all()
    summary_out = {}
    for model_id, s in summaries.items():
        summary_out[model_id] = {
            "overall_status":    s.overall_status.value,
            "critical_features": s.critical_features,
            "retrain_triggered": s.retrain_triggered,
        }
        log.info("[%s] drift=%s critical=%s retrain=%s",
                 model_id, s.overall_status.value, s.critical_features, s.retrain_triggered)
    ctx["ti"].xcom_push(key="drift_summaries", value=summary_out)
    return summary_out


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Check challenger & trigger A/B test
# ─────────────────────────────────────────────────────────────────────────────

def _check_challenger_and_ab(**ctx: Any) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, "/opt/analytica/src")
    from ml.ab_test import ABTestConfig, ABTestFramework  # type: ignore
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    triggered = []

    for model_id in ALL_MODELS:
        challenger_name = f"{model_id}_challenger"
        try:
            staging = client.get_latest_versions(challenger_name, stages=["Staging"])
            if not staging:
                continue
            log.info("[%s] Challenger found in Staging — running A/B evaluation", model_id)
            config = ABTestConfig(
                champion_model_id=model_id,
                challenger_model_id=challenger_name,
            )
            fw     = ABTestFramework(config)
            result = fw.evaluate()
            if result:
                triggered.append({
                    "model_id":    model_id,
                    "promoted":    result.promote,
                    "improvement": result.rmse_improvement_pct,
                })
                log.info("[%s] A/B promote=%s improvement=%.2f%%", model_id, result.promote, result.rmse_improvement_pct)
        except Exception as exc:
            log.warning("[%s] A/B check failed: %s", model_id, exc)

    ctx["ti"].xcom_push(key="ab_results", value=triggered)
    return {"ab_triggers": triggered}


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Push summary metrics to Pushgateway
# ─────────────────────────────────────────────────────────────────────────────

def _push_summary_to_pushgateway(**ctx: Any) -> None:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    registry = CollectorRegistry()
    health_gauge = Gauge(
        "tropi_model_serving_health_check_last_run",
        "Timestamp of last successful health check run",
        registry=registry,
    )
    stale_gauge = Gauge(
        "tropi_model_serving_stale_count",
        "Number of stale models at last health check",
        registry=registry,
    )
    drift_critical = Gauge(
        "tropi_model_drift_critical_count",
        "Number of models with CRITICAL drift at last check",
        registry=registry,
    )

    now           = datetime.now(timezone.utc).timestamp()
    stale_models  = ctx["ti"].xcom_pull(task_ids="check_model_staleness",   key="stale_models")  or []
    drift_sums    = ctx["ti"].xcom_pull(task_ids="run_drift_monitor",        key="drift_summaries") or {}
    n_critical    = sum(1 for s in drift_sums.values() if s.get("overall_status") == "CRITICAL")

    health_gauge.set(now)
    stale_gauge.set(len(stale_models))
    drift_critical.set(n_critical)

    try:
        push_to_gateway(PUSHGATEWAY_URL, job="analytica_model_health", registry=registry)
        log.info("Summary metrics pushed to Pushgateway (stale=%d, drift_critical=%d)",
                 len(stale_models), n_critical)
    except Exception as exc:
        log.warning("Pushgateway push failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Failure metric helper
# ─────────────────────────────────────────────────────────────────────────────

def _push_failure_metric(stage: str, reason: str) -> None:
    """Emit tropi_model_serving_health_dag_failure for Prometheus alerting."""
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        reg = CollectorRegistry()
        g   = Gauge(
            "tropi_model_serving_health_dag_failure",
            "Set to 1 when model_serving_health_check DAG fails",
            ["stage", "reason"],
            registry=reg,
        )
        g.labels(stage=stage, reason=reason[:120]).set(1)
        push_to_gateway(PUSHGATEWAY_URL, job="analytica_health_failure", registry=reg)
    except Exception:
        pass  # Best-effort; never swallow the original exception


def on_failure_callback(context: Dict[str, Any]) -> None:
    _push_failure_metric(
        stage=str(context.get("task_instance", {}).task_id if hasattr(context.get("task_instance", {}), "task_id") else "unknown"),
        reason=str(context.get("exception", "unknown")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="model_serving_health_check",
    description=(
        "ANALYTICA Sprint 3 daily model serving health check. "
        "Pings inference API, checks staleness, runs drift monitor, "
        "evaluates A/B challengers, pushes summary to Pushgateway. "
        "SLA: 10 minutes. Failure → tropi_model_serving_health_dag_failure metric."
    ),
    schedule_interval="0 6 * * *",
    start_date=days_ago(1),
    default_args={**DEFAULT_ARGS, "on_failure_callback": on_failure_callback},
    catchup=False,
    max_active_runs=1,
    tags=["analytica", "health", "drift", "ab-test", "sprint3"],
    dagrun_timeout=timedelta(minutes=10),
) as dag:

    t1 = PythonOperator(
        task_id="ping_health_endpoint",
        python_callable=_ping_health_endpoint,
        doc_md="GET /health on inference API. Fails DAG if HTTP status != 200.",
    )
    t2 = PythonOperator(
        task_id="check_model_staleness",
        python_callable=_check_model_staleness,
        doc_md="Check Prometheus for each model's last inference timestamp. Alert if any stale > 24 h.",
    )
    t3 = PythonOperator(
        task_id="run_drift_monitor",
        python_callable=_run_drift_monitor,
        doc_md="Run src/ml/drift_monitor.py for all 4 models. Log DriftReports to MLflow + Pushgateway.",
    )
    t4 = PythonOperator(
        task_id="check_challenger_and_ab",
        python_callable=_check_challenger_and_ab,
        doc_md="Scan MLflow Staging for challenger models. Trigger A/B evaluation if found. Auto-promote on criteria.",
    )
    t5 = PythonOperator(
        task_id="push_summary_to_pushgateway",
        python_callable=_push_summary_to_pushgateway,
        trigger_rule="all_done",  # always run even if upstream raises SLA warning
        doc_md="Push aggregate health/drift summary metrics to Prometheus Pushgateway.",
    )

    t1 >> t2 >> t3 >> t4 >> t5
