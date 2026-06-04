"""
ANALYTICA Airflow Retraining DAG — Sprint 2
workspace/src/airflow/retraining_dag.py

Defines four independent retraining DAGs (one per model), each wiring
retraining_pipeline.run_scheduled_retrain() to an Airflow PythonOperator
with an SNS on_failure_callback and a downstream Slack notification stub.

DAGs defined in this file
─────────────────────────
  analytica_retrain_xgb_precip       — weekly     Mon  18:00 UTC  (0 18 * * 1)
  analytica_retrain_prophet_seasonal — monthly    1st  19:00 UTC  (0 19 1 * *)
  analytica_retrain_cnn_landcover    — quarterly  1st  20:00 UTC  (0 20 1 */3 *)
  analytica_retrain_lstm_streamflow  — monthly    1st  19:00 UTC  (0 19 1 * *)

Each DAG topology:
  [retrain_{model}]  →  [mlops_slack_alert_{model}]
     PythonOperator          SlackWebhookOperator (stub)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import boto3
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Airflow Variable keys  (configure in Airflow UI or via CLI export)
# ─────────────────────────────────────────────────────────────────────────────
VAR_SNS_TOPIC_ARN   = "ANALYTICA_SNS_FAILURE_TOPIC_ARN"
VAR_SNS_REGION      = "ANALYTICA_SNS_REGION"
VAR_SLACK_WEBHOOK   = "ANALYTICA_SLACK_MLOPS_WEBHOOK_URL"   # mlops-alerts channel
VAR_MLFLOW_URI      = "ANALYTICA_MLFLOW_TRACKING_URI"


def _var(key: str, fallback: str = "") -> str:
    try:
        return Variable.get(key, default_var=fallback)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# SNS on_failure_callback
# ─────────────────────────────────────────────────────────────────────────────

def sns_failure_callback(context: Dict[str, Any]) -> None:
    """
    Publish a structured failure notification to the configured SNS topic.
    Called automatically by Airflow when any task in the DAG fails.
    Fails silently if SNS is not configured (non-blocking).
    """
    topic_arn = _var(VAR_SNS_TOPIC_ARN)
    region    = _var(VAR_SNS_REGION, "ap-southeast-3")   # AWS Jakarta region

    if not topic_arn:
        log.warning("SNS topic ARN not configured — skipping failure alert")
        return

    dag_id   = context.get("dag").dag_id       if context.get("dag")  else "unknown"
    task_id  = context.get("task_instance").task_id if context.get("task_instance") else "unknown"
    run_id   = context.get("run_id", "unknown")
    exc      = context.get("exception", "No exception captured")

    message = {
        "agent":          "ANALYTICA",
        "event":          "task_failure",
        "dag_id":         dag_id,
        "task_id":        task_id,
        "run_id":         run_id,
        "execution_date": str(context.get("execution_date", "")),
        "exception":      str(exc)[:1000],
        "log_url":        context.get("task_instance").log_url
                          if context.get("task_instance") else "",
        "alerted_at":     datetime.utcnow().isoformat() + "Z",
    }

    try:
        sns = boto3.client("sns", region_name=region)
        sns.publish(
            TopicArn = topic_arn,
            Subject  = f"[ANALYTICA] DAG failure: {dag_id} / {task_id}",
            Message  = json.dumps(message, indent=2),
            MessageAttributes={
                "agent":    {"DataType": "String", "StringValue": "ANALYTICA"},
                "dag_id":   {"DataType": "String", "StringValue": dag_id},
                "severity": {"DataType": "String", "StringValue": "ERROR"},
            },
        )
        log.info(f"SNS failure alert published for {dag_id}/{task_id}")
    except Exception as exc_inner:
        log.error(f"Failed to publish SNS alert (non-fatal): {exc_inner}")


# ─────────────────────────────────────────────────────────────────────────────
# Default DAG arguments
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARGS: Dict[str, Any] = {
    "owner":               "ANALYTICA",
    "depends_on_past":     False,
    "email":               ["analytica@tropiclimate.id"],
    "email_on_failure":    True,
    "email_on_retry":      False,
    "retries":             2,
    "retry_delay":         timedelta(minutes=10),
    "execution_timeout":   timedelta(hours=6),
    "on_failure_callback": sns_failure_callback,
}


# ─────────────────────────────────────────────────────────────────────────────
# Retraining callable  (PythonOperator target)
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduled_retrain(model_type: str, **context) -> Dict[str, Any]:
    """
    Entry point called by every retraining PythonOperator.
    Delegates to retraining_pipeline.run_scheduled_retrain() and
    pushes the result summary to XCom for the downstream Slack task.
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    try:
        from analytics.retraining_pipeline import run_scheduled_retrain as _run
        result = _run(
            model_type   = model_type,
            run_id       = context.get("run_id"),
            execution_dt = str(context.get("execution_date", "")),
            mlflow_uri   = _var(VAR_MLFLOW_URI, "http://localhost:5000"),
        )
    except ImportError:
        # Graceful degradation when retraining_pipeline is not yet installed
        log.warning("retraining_pipeline not importable — running stub")
        result = {
            "model_type":  model_type,
            "status":      "stub_run",
            "run_id":      context.get("run_id"),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }

    context["ti"].xcom_push(key="retrain_result", value=result)
    log.info(f"[{model_type}] Retraining complete: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Slack notification callable  (SlackOperator stub)
# ─────────────────────────────────────────────────────────────────────────────

def slack_mlops_alert(model_type: str, **context) -> None:
    """
    Downstream Slack notification stub — posts retraining outcome to
    the #mlops-alerts channel via incoming webhook.

    STUB: replace the HTTP call with SlackWebhookOperator or use the
    Airflow Slack provider (apache-airflow-providers-slack) when the
    Slack connection 'slack_mlops_alerts' is configured in Airflow Connections.

    Example production replacement:
        from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
        task = SlackWebhookOperator(
            task_id             = "mlops_slack_alert",
            slack_webhook_conn_id = "slack_mlops_alerts",
            message             = message,
            channel             = "#mlops-alerts",
        )
    """
    import requests

    ti           = context["ti"]
    retrain_res  = ti.xcom_pull(task_ids=f"retrain_{model_type}", key="retrain_result") or {}
    webhook_url  = _var(VAR_SLACK_WEBHOOK)

    status_emoji = {
        "promoted":    ":white_check_mark:",
        "archived":    ":archive:",
        "no_op":       ":zzz:",
        "stub_run":    ":construction:",
    }.get(retrain_res.get("action", retrain_res.get("status", "")), ":information_source:")

    message = (
        f"{status_emoji} *ANALYTICA Retraining — `{model_type}`*\n"
        f">*Status:* `{retrain_res.get('action', retrain_res.get('status', 'unknown'))}`\n"
        f">*Run ID:* `{retrain_res.get('run_id', 'n/a')}`\n"
        f">*New production version:* `{retrain_res.get('new_production_v', 'n/a')}`\n"
        f">*Drift detected:* `{retrain_res.get('drift_detected', 'n/a')}`\n"
        f">*A/B win rate:* `{retrain_res.get('ab_win_rate', 'n/a')}`\n"
        f">*DAG run:* `{context.get('run_id', 'n/a')}`"
    )

    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json    = {"text": message, "channel": "#mlops-alerts"},
                timeout = 10,
            )
            resp.raise_for_status()
            log.info(f"[{model_type}] Slack mlops-alerts notified: HTTP {resp.status_code}")
        except Exception as exc:
            log.warning(f"[{model_type}] Slack notification failed (non-fatal): {exc}")
    else:
        log.info(
            f"[{model_type}] ANALYTICA_SLACK_MLOPS_WEBHOOK_URL not set "
            f"— Slack stub skipped. Message would have been:\n{message}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DAG factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_dag(
    model_type:        str,
    schedule_interval: str,
    description:       str,
) -> DAG:
    """Build a two-task retraining DAG: retrain → slack_alert."""
    with DAG(
        dag_id            = f"analytica_retrain_{model_type}",
        description       = description,
        default_args      = DEFAULT_ARGS,
        schedule_interval = schedule_interval,
        start_date        = datetime(2026, 6, 1),
        catchup           = False,
        max_active_runs   = 1,
        tags              = ["analytica", "retraining", "mlops", model_type],
    ) as dag:

        t_retrain = PythonOperator(
            task_id         = f"retrain_{model_type}",
            python_callable = run_scheduled_retrain,
            op_kwargs       = {"model_type": model_type},
            doc_md          = (
                f"Train challenger for `{model_type}`, run A/B evaluation, "
                "promote or archive via MLflow registry."
            ),
        )

        # ── Slack stub ── (swap for SlackWebhookOperator when Airflow
        #                   Slack provider + connection are configured)
        t_slack = PythonOperator(
            task_id         = f"mlops_slack_alert_{model_type}",
            python_callable = slack_mlops_alert,
            op_kwargs       = {"model_type": model_type},
            trigger_rule    = "all_done",   # runs even if retrain fails
            doc_md          = (
                "Post retraining outcome to #mlops-alerts Slack channel. "
                "STUB — replace with SlackWebhookOperator when connection is ready."
            ),
        )

        t_retrain >> t_slack

    return dag


# ─────────────────────────────────────────────────────────────────────────────
# Instantiate DAGs  (Airflow discovers module-level DAG objects)
# ─────────────────────────────────────────────────────────────────────────────

dag_xgb_precip = _build_dag(
    model_type        = "xgb_precip",
    schedule_interval = "0 18 * * 1",    # weekly — every Monday 18:00 UTC
    description       = (
        "ANALYTICA — Weekly XGBoost precipitation nowcast retraining. "
        "Fires every Monday 18:00 UTC. Drift: PSI > 0.2 or KS p < 0.05."
    ),
)

dag_prophet_seasonal = _build_dag(
    model_type        = "prophet_seasonal",
    schedule_interval = "0 19 1 * *",    # monthly — 1st of month 19:00 UTC
    description       = (
        "ANALYTICA — Monthly Prophet seasonal forecast retraining. "
        "Fires on the 1st of each month at 19:00 UTC."
    ),
)

dag_cnn_landcover = _build_dag(
    model_type        = "cnn_landcover",
    schedule_interval = "0 20 1 */3 *",  # quarterly — 1st Jan/Apr/Jul/Oct 20:00 UTC
    description       = (
        "ANALYTICA — Quarterly ResNet+Attention CNN land-cover retraining. "
        "Fires 1 Jan / 1 Apr / 1 Jul / 1 Oct at 20:00 UTC."
    ),
)

dag_lstm_streamflow = _build_dag(
    model_type        = "lstm_streamflow",
    schedule_interval = "0 19 1 * *",    # monthly — 1st of month 19:00 UTC
    description       = (
        "ANALYTICA — Monthly LSTM streamflow retraining. "
        "Fires on the 1st of each month at 19:00 UTC."
    ),
)
