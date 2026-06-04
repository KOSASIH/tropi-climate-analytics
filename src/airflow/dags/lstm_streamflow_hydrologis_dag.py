"""
DAG: analytica_lstm_streamflow_hydrologis
ANALYTICA — LSTM Streamflow model wired to HYDROLOGIS weight delivery.

Triggered: monthly on the 1st at 19:30 WIB / 12:30 UTC (30 min after
HYDROLOGIS weight delivery window closes at 19:00 WIB).

DAG topology:
  [check_weight_delivery]     ShortCircuitOperator
           │  (short-circuits if no new weights)
           ▼
  [receive_hydrologis_weights] PythonOperator — ingest + MLflow register
           │
           ▼
  [retrain_lstm_streamflow]   PythonOperator — warm-start retrain
           │
           ▼
  [evaluate_and_promote]      PythonOperator — A/B vs champion
           │
           ▼
  [generate_model_card]       PythonOperator — JSON + Markdown model card
           │
           ▼
  [notify_climate_os]         PythonOperator — POST to CLIMATE-OS webhook
           │
           ▼
  [mlops_slack_alert]         PythonOperator — #mlops-alerts stub
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from analytica.airflow.hydrologis_weight_delivery import (  # type: ignore[import]
    receive_hydrologis_weights,
    weight_delivery_available,
)
from analytica.airflow.dag_factory import (  # type: ignore[import]
    sns_failure_callback,
    slack_mlops_alert,
    _var,
)

log = logging.getLogger(__name__)

MODEL_TYPE = "lstm_streamflow"

# ─────────────────────────────────────────────────────────────────────────────
# Default args
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARGS: Dict[str, Any] = {
    "owner":               "ANALYTICA",
    "depends_on_past":     False,
    "email":               ["analytica@tropiclimate.id"],
    "email_on_failure":    True,
    "email_on_retry":      False,
    "retries":             2,
    "retry_delay":         timedelta(minutes=15),
    "execution_timeout":   timedelta(hours=5),
    "on_failure_callback": sns_failure_callback,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage callables
# ─────────────────────────────────────────────────────────────────────────────

def retrain_lstm_streamflow(**context) -> Dict[str, Any]:
    """
    Warm-start retrain the LSTM streamflow model using weights delivered
    by HYDROLOGIS.  Pushes retrain_result to XCom.
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    weight_result = context["ti"].xcom_pull(
        task_ids="receive_hydrologis_weights", key="weight_delivery_result"
    ) or {}

    mlflow_uri = _var("ANALYTICA_MLFLOW_TRACKING_URI", "http://localhost:5000")

    try:
        from analytics.retraining_pipeline import run_scheduled_retrain as _run
        result = _run(
            model_type   = MODEL_TYPE,
            run_id       = context.get("run_id"),
            execution_dt = str(context.get("execution_date", "")),
            mlflow_uri   = mlflow_uri,
            extra_kwargs = {
                "warm_start_weights_dir": weight_result.get("weights_dir"),
                "hydrologis_version":     weight_result.get("hydrologis_version"),
                "hydrologis_run_id":      weight_result.get("hydrologis_run_id"),
                "feature_order":          weight_result.get("feature_order"),
                "transfer_learning":      True,
            },
        )
    except (ImportError, TypeError):
        log.warning("retraining_pipeline not importable / extra_kwargs unsupported "
                    "— running stub")
        result = {
            "model_type":            MODEL_TYPE,
            "status":                "stub_retrain",
            "hydrologis_version":    weight_result.get("hydrologis_version", "unknown"),
            "transfer_learning":     True,
            "run_id":                context.get("run_id"),
            "completed_at":         datetime.utcnow().isoformat() + "Z",
        }

    context["ti"].xcom_push(key="retrain_result", value=result)
    log.info(f"[{MODEL_TYPE}] Warm-start retrain complete: {result}")
    return result


def evaluate_and_promote(**context) -> Dict[str, Any]:
    """
    A/B evaluation of new LSTM challenger vs champion, then promote or archive.
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    retrain = context["ti"].xcom_pull(
        task_ids="retrain_lstm_streamflow", key="retrain_result"
    ) or {}

    if retrain.get("status") == "stub_retrain":
        result = {"action": "stub_no_op", "reason": "stub_retrain_mode"}
        context["ti"].xcom_push(key="promotion_result", value=result)
        return result

    try:
        from analytics.retraining_pipeline import ChampionChallengerManager
        from analytics.mlflow_setup import MODEL_REGISTRY_NAMES

        model_name = MODEL_REGISTRY_NAMES.get(MODEL_TYPE, MODEL_TYPE)
        cc = ChampionChallengerManager(model_name)
        cc.champion_v   = retrain.get("champion_version")
        cc.challenger_v = retrain.get("run_id")

        champ_rmse      = retrain.get("champion_metrics", {}).get("val_rmse", 9999.0)
        challenger_rmse = retrain.get("challenger_metrics", {}).get("val_rmse", 9999.0)

        for _ in range(5):
            cc.evaluate_round(champ_rmse, challenger_rmse,
                              metric_name="val_rmse", lower_is_better=True)

        result = cc.finalize()
    except ImportError:
        result = {"action": "skipped", "reason": "pipeline_unavailable"}

    context["ti"].xcom_push(key="promotion_result", value=result)
    log.info(f"[{MODEL_TYPE}] Promotion result: {result}")
    return result


def generate_model_card_lstm(**context) -> Dict[str, Any]:
    """Generate model card for the retrained LSTM streamflow model."""
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    try:
        from analytics.explainability import ModelCardGenerator

        retrain    = context["ti"].xcom_pull(
            task_ids="retrain_lstm_streamflow", key="retrain_result") or {}
        promotion  = context["ti"].xcom_pull(
            task_ids="evaluate_and_promote",    key="promotion_result") or {}
        weight_res = context["ti"].xcom_pull(
            task_ids="receive_hydrologis_weights", key="weight_delivery_result") or {}

        gen   = ModelCardGenerator(
            MODEL_TYPE,
            str(promotion.get("new_production_v", "unknown")),
            retrain.get("run_id", "unknown"),
            retrain.get("challenger_metrics", {}),
        )
        paths = gen.save(output_dir="/opt/analytica/docs/model_cards")
        paths["hydrologis_version"] = weight_res.get("hydrologis_version")
        context["ti"].xcom_push(key="model_card_paths", value=paths)
        log.info(f"[{MODEL_TYPE}] Model card written: {paths}")
        return paths

    except ImportError:
        log.warning("explainability module not available — skipping model card")
        result = {"skipped": True}
        context["ti"].xcom_push(key="model_card_paths", value=result)
        return result


def notify_climate_os_lstm(**context) -> Dict[str, Any]:
    """POST LSTM retraining outcome to CLIMATE-OS webhook."""
    import requests

    ti = context["ti"]
    weight_res = ti.xcom_pull(task_ids="receive_hydrologis_weights",
                              key="weight_delivery_result") or {}
    retrain    = ti.xcom_pull(task_ids="retrain_lstm_streamflow",
                              key="retrain_result")           or {}
    promotion  = ti.xcom_pull(task_ids="evaluate_and_promote",
                              key="promotion_result")         or {}
    card_paths = ti.xcom_pull(task_ids="generate_model_card",
                              key="model_card_paths")         or {}

    payload = {
        "agent":                  "ANALYTICA",
        "sprint":                 2,
        "model_type":             MODEL_TYPE,
        "dag_run_id":             context["run_id"],
        "execution_date":        str(context["execution_date"]),
        "hydrologis_version":     weight_res.get("hydrologis_version"),
        "hydrologis_run_id":      weight_res.get("hydrologis_run_id"),
        "transfer_learning":      weight_res.get("hydrologis_version") is not None,
        "challenger_run_id":      retrain.get("run_id"),
        "challenger_metrics":     retrain.get("challenger_metrics", {}),
        "promotion_action":       promotion.get("action"),
        "new_production_v":       promotion.get("new_production_v"),
        "model_card":             card_paths,
        "summary_at":             datetime.utcnow().isoformat() + "Z",
    }

    webhook_url = _var("ANALYTICA_CLIMATE_OS_WEBHOOK_URL")
    if webhook_url:
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            log.info(f"[{MODEL_TYPE}] CLIMATE-OS notified: HTTP {resp.status_code}")
        except Exception as exc:
            log.warning(f"[{MODEL_TYPE}] Webhook failed (non-fatal): {exc}")
    else:
        log.info(f"[{MODEL_TYPE}] No webhook URL — CLIMATE-OS notification logged only\n"
                 + json.dumps(payload, indent=2))

    context["ti"].xcom_push(key="retrain_result", value=payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id            = "analytica_lstm_streamflow_hydrologis",
    description       = (
        "ANALYTICA — LSTM streamflow retraining triggered by HYDROLOGIS weight delivery. "
        "Monthly on 1st at 12:30 UTC (19:30 WIB). "
        "Short-circuits if no new weights. Warm-start transfer learning."
    ),
    default_args      = DEFAULT_ARGS,
    schedule_interval = "30 12 1 * *",    # 1st of month 12:30 UTC = 19:30 WIB
    start_date        = datetime(2026, 6, 1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["analytica", "hydrologis", "lstm", "streamflow",
                         "transfer-learning", "retraining"],
) as dag:

    t_check = ShortCircuitOperator(
        task_id         = "check_weight_delivery",
        python_callable = weight_delivery_available,
        doc_md          = (
            "Probe HYDROLOGIS drop-zone for new weight manifest. "
            "Short-circuits (skips remaining tasks) if no new weights are available."
        ),
    )

    t_receive = PythonOperator(
        task_id         = "receive_hydrologis_weights",
        python_callable = receive_hydrologis_weights,
        doc_md          = (
            "Download + validate HYDROLOGIS weight artifacts. "
            "Verifies SHA-256 checksum and registers in MLflow."
        ),
    )

    t_retrain = PythonOperator(
        task_id         = "retrain_lstm_streamflow",
        python_callable = retrain_lstm_streamflow,
        doc_md          = (
            "Warm-start LSTM streamflow model retrain using HYDROLOGIS weights "
            "(transfer learning). Covers Ciliwung, Brantas, Solo rivers."
        ),
    )

    t_evaluate = PythonOperator(
        task_id         = "evaluate_and_promote",
        python_callable = evaluate_and_promote,
        doc_md          = "A/B champion vs challenger evaluation (5 rounds, val_rmse).",
    )

    t_card = PythonOperator(
        task_id         = "generate_model_card",
        python_callable = generate_model_card_lstm,
        doc_md          = "Generate JSON + Markdown model card (Model Card 2.0).",
    )

    t_notify_cos = PythonOperator(
        task_id         = "notify_climate_os",
        python_callable = notify_climate_os_lstm,
        trigger_rule    = "all_done",
        doc_md          = "POST retraining summary to CLIMATE-OS webhook.",
    )

    t_slack = PythonOperator(
        task_id         = f"mlops_slack_alert_{MODEL_TYPE}",
        python_callable = slack_mlops_alert,
        op_kwargs       = {"model_type": MODEL_TYPE},
        trigger_rule    = "all_done",
        doc_md          = (
            "#mlops-alerts Slack notification stub. "
            "Replace with SlackWebhookOperator when slack_mlops_alerts connection is set."
        ),
    )

    # ── Topology ──────────────────────────────────────────────────────────
    (t_check
     >> t_receive
     >> t_retrain
     >> t_evaluate
     >> t_card
     >> t_notify_cos
     >> t_slack)
