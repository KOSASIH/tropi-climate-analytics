"""
Airflow DAG — Emergency Flood Alert (TriggerDagRunOperator target)
HYDROLOGIS Sprint 5 | Deliverable 4

dag_id:   emergency_flood_alert_dag
schedule: None — triggered exclusively by flood_early_warning_dag
          BranchPythonOperator when EMERGENCY stage detected
SLA:      3 minutes

Expected conf keys:
  river_id                str    e.g. "ciliwung"
  flood_stage             str    "WARNING" | "EMERGENCY"
  peak_q_m3s              float  peak discharge (m³/s)
  affected_villages       list   village names (may be empty list)
  forecast_horizon_hours  int    6 | 12 | 24

Task graph:
  parse_alert_conf
       ↓
  send_bmkg_notification  (FloodAlertNotifier.send_flood_alert)
       ↓
  write_visualia_payload  (confirm JSON artifact path)
       ↓
  log_mlflow_alert        (experiment=flood_alerts)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

WORKSPACE  = os.getenv("WORKSPACE_ROOT", "workspace")
MLFLOW_EXP = "flood_alerts"

default_args = {
    "owner":             "hydrologis",
    "depends_on_past":   False,
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=3),
}

# ---------------------------------------------------------------------------
# Task 1: parse_alert_conf
# ---------------------------------------------------------------------------

def parse_alert_conf(**context) -> dict:
    """
    Extract and validate alert parameters from DAG run conf.
    Provides safe defaults for any missing keys so downstream tasks never crash.
    """
    conf = context.get("dag_run").conf or {}

    river_id               = str(conf.get("river_id",               "unknown"))
    flood_stage            = str(conf.get("flood_stage",            "EMERGENCY"))
    peak_q_m3s             = float(conf.get("peak_q_m3s",           0.0))
    affected_villages      = list(conf.get("affected_villages",     []))
    forecast_horizon_hours = int(conf.get("forecast_horizon_hours", 6))

    # Derive affected_villages from river-specific defaults if conf sent empty list
    if not affected_villages:
        defaults = {
            "ciliwung": ["Kampung Melayu", "Bukit Duri", "Pengadegan", "Cawang", "Rawajati"],
            "brantas":  ["Kediri Kota", "Tulungagung", "Blitar", "Mojokerto"],
            "solo":     ["Surakarta", "Demak", "Grobogan", "Bojonegoro"],
        }
        affected_villages = defaults.get(river_id, [])

    parsed = {
        "river_id":               river_id,
        "flood_stage":            flood_stage,
        "peak_q_m3s":             peak_q_m3s,
        "affected_villages":      affected_villages,
        "forecast_horizon_hours": forecast_horizon_hours,
        "breach_time_utc":        conf.get("breach_time_utc", datetime.now(timezone.utc).isoformat()),
    }

    logger.warning(
        "Emergency flood alert triggered | river=%s stage=%s peak=%.1f m³/s horizon=%dh villages=%d",
        river_id, flood_stage, peak_q_m3s, forecast_horizon_hours, len(affected_villages),
    )
    return parsed


# ---------------------------------------------------------------------------
# Task 2: send_bmkg_notification
# ---------------------------------------------------------------------------

def send_bmkg_notification(**context) -> dict:
    """
    Invoke FloodAlertNotifier.send_flood_alert — delivers to BMKG webhook + VISUALIA JSON.
    """
    from src.hydrology.flood_alert_notifier import FloodAlertNotifier

    ti     = context["ti"]
    params = ti.xcom_pull(task_ids="parse_alert_conf")

    breach_time = datetime.fromisoformat(params["breach_time_utc"])
    notifier    = FloodAlertNotifier(workspace=WORKSPACE)

    result = notifier.send_flood_alert(
        river_id               = params["river_id"],
        flood_stage            = params["flood_stage"],
        peak_q_m3s             = params["peak_q_m3s"],
        affected_villages      = params["affected_villages"],
        forecast_horizon_hours = params["forecast_horizon_hours"],
        breach_time            = breach_time,
    )

    logger.info(
        "Alert delivery complete | id=%s bmkg_ok=%s visualia_path=%s",
        result.alert_id, result.bmkg_webhook_ok, result.visualia_path,
    )

    if not result.success:
        logger.error(
            "Alert delivery partial failure | bmkg_ok=%s visualia_path=%s",
            result.bmkg_webhook_ok, result.visualia_path,
        )

    return {
        "alert_id":        result.alert_id,
        "bmkg_webhook_ok": result.bmkg_webhook_ok,
        "visualia_path":   result.visualia_path,
        "success":         result.success,
        "river_id":        params["river_id"],
        "flood_stage":     params["flood_stage"],
    }


# ---------------------------------------------------------------------------
# Task 3: write_visualia_payload  (confirmation / idempotency check)
# ---------------------------------------------------------------------------

def write_visualia_payload(**context) -> str:
    """
    Confirm VISUALIA JSON artifact exists (written by FloodAlertNotifier).
    If missing (notifier failed), write a minimal fallback artifact.
    """
    ti       = context["ti"]
    delivery = ti.xcom_pull(task_ids="send_bmkg_notification")
    params   = ti.xcom_pull(task_ids="parse_alert_conf")

    visualia_path = delivery.get("visualia_path")

    if visualia_path and os.path.exists(visualia_path):
        logger.info("VISUALIA payload confirmed: %s", visualia_path)
        return visualia_path

    # Fallback: write minimal payload
    alerts_dir = os.path.join(WORKSPACE, "output", "alerts")
    os.makedirs(alerts_dir, exist_ok=True)
    ts_str     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fallback_path = os.path.join(alerts_dir, f"flood_{params['river_id']}_{ts_str}_fallback.json")

    payload = {
        "alert_id":               delivery.get("alert_id", f"FLOOD-{params['river_id']}-fallback"),
        "river_id":               params["river_id"],
        "flood_stage":            params["flood_stage"],
        "peak_discharge_m3s":     params["peak_q_m3s"],
        "affected_villages":      params["affected_villages"],
        "forecast_horizon_hours": params["forecast_horizon_hours"],
        "breach_time_utc":        params["breach_time_utc"],
        "issued_at_utc":          datetime.now(timezone.utc).isoformat(),
        "source":                 "emergency_flood_alert_dag/fallback",
        "bmkg_webhook_delivered": delivery.get("bmkg_webhook_ok", False),
    }

    with open(fallback_path, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    logger.warning("VISUALIA fallback payload written: %s", fallback_path)
    return fallback_path


# ---------------------------------------------------------------------------
# Task 4: log_mlflow_alert
# ---------------------------------------------------------------------------

def log_mlflow_alert(**context) -> None:
    """
    Log flood alert to MLflow experiment 'flood_alerts'.
    Tags: river_id, flood_stage. Metrics: peak_q_m3s, horizon_hours, villages_count.
    """
    try:
        import mlflow

        ti       = context["ti"]
        params   = ti.xcom_pull(task_ids="parse_alert_conf")
        delivery = ti.xcom_pull(task_ids="send_bmkg_notification")

        mlflow.set_experiment(MLFLOW_EXP)
        with mlflow.start_run(
            run_name=f"flood_alert_{params['river_id']}_{params['flood_stage']}",
            tags={
                "dag":         "emergency_flood_alert_dag",
                "river_id":    params["river_id"],
                "flood_stage": params["flood_stage"],
                "agent":       "HYDROLOGIS",
            },
        ):
            mlflow.log_metric("peak_q_m3s",             params["peak_q_m3s"])
            mlflow.log_metric("forecast_horizon_hours",  params["forecast_horizon_hours"])
            mlflow.log_metric("affected_villages_count", len(params["affected_villages"]))
            mlflow.log_metric("bmkg_webhook_ok",         int(delivery.get("bmkg_webhook_ok", False)))
            mlflow.log_param("alert_id",                 delivery.get("alert_id", ""))
            mlflow.log_param("breach_time_utc",          params["breach_time_utc"])

            # Log VISUALIA artifact if available
            visualia_path = ti.xcom_pull(task_ids="write_visualia_payload")
            if visualia_path and os.path.exists(visualia_path):
                mlflow.log_artifact(visualia_path, artifact_path="flood_alerts")

        logger.info(
            "MLflow alert logged | river=%s stage=%s exp=%s",
            params["river_id"], params["flood_stage"], MLFLOW_EXP,
        )
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="emergency_flood_alert_dag",
    description="Emergency flood alert delivery — triggered by flood_early_warning_30min on EMERGENCY stage",
    schedule=None,               # Triggered only via TriggerDagRunOperator
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["hydrologis", "flood", "emergency", "alert"],
    doc_md="""
## HYDROLOGIS — Emergency Flood Alert DAG

**Schedule:** None — triggered by `flood_early_warning_30min` BranchPythonOperator  
**SLA:** 3 minutes  
**Triggered when:** Any river reaches EMERGENCY stage in flood inundation forecast

**Conf keys:** `river_id`, `flood_stage`, `peak_q_m3s`, `affected_villages`, `forecast_horizon_hours`

**Alert channels:**
1. BMKG webhook (`BMKG_WEBHOOK_URL` env, timeout 8s, retry 3×)
2. VISUALIA JSON → `workspace/output/alerts/flood_{river_id}_{timestamp}.json`

**MLflow experiment:** `flood_alerts` | tags: `river_id`, `flood_stage`
    """,
    dagrun_timeout=timedelta(minutes=3),
) as dag:

    t_parse = PythonOperator(
        task_id="parse_alert_conf",
        python_callable=parse_alert_conf,
    )

    t_notify = PythonOperator(
        task_id="send_bmkg_notification",
        python_callable=send_bmkg_notification,
    )

    t_visualia = PythonOperator(
        task_id="write_visualia_payload",
        python_callable=write_visualia_payload,
    )

    t_mlflow = PythonOperator(
        task_id="log_mlflow_alert",
        python_callable=log_mlflow_alert,
    )

    t_parse >> t_notify >> t_visualia >> t_mlflow
