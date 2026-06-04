"""
Airflow DAG — GRACE-FO Aquifer Depletion Tracker (Monthly)
HYDROLOGIS Sprint 4 | Deliverable 1

dag_id:   hydrologis_aquifer_tracker
schedule: 0 6 3 * * (Asia/Jakarta) — 3rd of each month at 06:00 WIB
SLA:      15 minutes

Task graph:
  fetch_grace_data → run_aquifer_tracker → log_mlflow → check_depletion_alerts
                                                              │
                                               depletion_alert=True
                                                              ↓
                                          TriggerDagRunOperator → notify_stakeholders_dag
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("WORKSPACE_ROOT", "workspace")
MLFLOW_EXPERIMENT = "hydrologis_aquifer_tracker"

# ---------------------------------------------------------------------------
# Default DAG args
# ---------------------------------------------------------------------------

default_args = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=15),
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def fetch_grace_data(**context) -> dict:
    """
    Pull GRACE-FO RL06 Mascon data for the current data month.

    Production:
      - Check workspace/data/grace/ for cached monthly NetCDF
      - If missing, trigger DATA-FLOW GRACE pull (coordinate via shared Airflow pool)
      - Validate file presence and checksum before proceeding

    Sprint 4: confirms data dir exists; actual pull handled by DATA-FLOW pipeline.
    """
    logical_date = context["logical_date"]
    # Data month is the month BEFORE the run date (GRACE-FO data ~30-day lag)
    first_of_run_month = logical_date.replace(day=1)
    data_month = (first_of_run_month - timedelta(days=1)).replace(day=1)

    grace_dir = os.path.join(WORKSPACE, "data", "grace")
    os.makedirs(grace_dir, exist_ok=True)

    logger.info("GRACE-FO fetch | data_month=%s grace_dir=%s", data_month, grace_dir)
    return {"data_month": data_month.isoformat(), "grace_dir": grace_dir}


def run_aquifer_tracker(**context) -> dict:
    """
    Instantiate GRACEFOAquiferTracker and run .update_all_aquifers().
    Returns a dict with depletion_alert status per aquifer.
    """
    from src.hydrology.aquifer_tracker import GRACEFOAquiferTracker

    ti = context["ti"]
    upstream = ti.xcom_pull(task_ids="fetch_grace_data")
    data_month = date.fromisoformat(upstream["data_month"])

    tracker = GRACEFOAquiferTracker()
    # update_all_aquifers() delegates to run() for current data_month
    status = tracker.update_all_aquifers(data_month=data_month)

    logger.info(
        "Aquifer tracker complete | month=%s processed=%d alerts=%d critical=%d",
        data_month,
        status.aquifers_processed,
        len(status.aquifers_alert),
        len(status.aquifers_critical),
    )

    result = {
        "data_month":         data_month.isoformat(),
        "aquifers_processed": status.aquifers_processed,
        "aquifers_alert":     status.aquifers_alert,
        "aquifers_critical":  status.aquifers_critical,
        "output_paths":       status.output_paths,
        "has_depletion_alert": len(status.aquifers_alert) > 0,
    }
    return result


def log_to_mlflow(**context) -> None:
    """Log aquifer depletion metrics to MLflow experiment."""
    try:
        import mlflow

        ti = context["ti"]
        tracker_result = ti.xcom_pull(task_ids="run_aquifer_tracker")
        data_month = tracker_result["data_month"]

        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(
            run_name=f"aquifer_tracker_{data_month}",
            tags={"dag": "hydrologis_aquifer_tracker", "agent": "HYDROLOGIS"},
        ):
            mlflow.log_metric("aquifers_processed", tracker_result["aquifers_processed"])
            mlflow.log_metric("aquifers_in_alert",  len(tracker_result["aquifers_alert"]))
            mlflow.log_metric("aquifers_critical",  len(tracker_result["aquifers_critical"]))
            mlflow.log_param("data_month", data_month)
            for path in tracker_result.get("output_paths", []):
                if os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path="aquifer_reports")
        logger.info("MLflow logging complete for aquifer tracker run %s", data_month)
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)


def check_depletion_alerts(**context) -> str:
    """
    BranchPythonOperator: route to alert trigger if any depletion_alert=True,
    otherwise route to skip_alert (no-op).
    """
    ti = context["ti"]
    result = ti.xcom_pull(task_ids="run_aquifer_tracker")
    if result.get("has_depletion_alert"):
        logger.warning(
            "Depletion alerts detected for: %s — triggering notify_stakeholders_dag",
            result.get("aquifers_alert"),
        )
        return "trigger_stakeholder_notification"
    return "skip_alert"


def build_stakeholder_conf(**context) -> dict:
    """Build conf payload for notify_stakeholders_dag trigger."""
    ti = context["ti"]
    result = ti.xcom_pull(task_ids="run_aquifer_tracker")
    return {
        "source_dag": "hydrologis_aquifer_tracker",
        "data_month": result.get("data_month"),
        "alert_type": "AQUIFER_DEPLETION",
        "aquifers_alert": result.get("aquifers_alert", []),
        "aquifers_critical": result.get("aquifers_critical", []),
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="hydrologis_aquifer_tracker",
    description="Monthly GRACE-FO aquifer depletion tracker for 5 critical Indonesian aquifer systems",
    schedule="0 6 3 * *",
    start_date=datetime(2026, 1, 3, tzinfo=None),
    catchup=False,
    default_args=default_args,
    tags=["hydrologis", "aquifer", "grace-fo", "monthly"],
    doc_md="""
## HYDROLOGIS — Aquifer Depletion Tracker (Monthly)

**Schedule:** 3rd of each month at 06:00 WIB (UTC+7)  
**SLA:** 15 minutes  
**Data source:** NASA GRACE-FO RL06 Mascon (CSR)  
**Aquifers:** North Jakarta · Bandung Basin · Semarang · Surabaya · Makassar  

**Alert routing:** Any `depletion_alert=True` triggers `notify_stakeholders_dag`  
**MLflow experiment:** `hydrologis_aquifer_tracker`
    """,
    dagrun_timeout=timedelta(minutes=15),
) as dag:

    t_fetch = PythonOperator(
        task_id="fetch_grace_data",
        python_callable=fetch_grace_data,
    )

    t_run = PythonOperator(
        task_id="run_aquifer_tracker",
        python_callable=run_aquifer_tracker,
    )

    t_mlflow = PythonOperator(
        task_id="log_mlflow",
        python_callable=log_to_mlflow,
    )

    t_branch = BranchPythonOperator(
        task_id="check_depletion_alerts",
        python_callable=check_depletion_alerts,
    )

    t_trigger = TriggerDagRunOperator(
        task_id="trigger_stakeholder_notification",
        trigger_dag_id="notify_stakeholders_dag",   # stub DAG — see dags/notify_stakeholders_dag.py
        conf={"alert_type": "AQUIFER_DEPLETION"},
        wait_for_completion=False,
        reset_dag_run=False,
    )

    t_skip = EmptyOperator(task_id="skip_alert")

    # Task dependencies
    t_fetch >> t_run >> t_mlflow >> t_branch >> [t_trigger, t_skip]
