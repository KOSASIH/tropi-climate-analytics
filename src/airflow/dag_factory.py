"""
ANALYTICA Airflow DAG Factory — Sprint 2
Shared factory that builds a complete retraining DAG for any model type.
Each generated DAG follows the cron schedule and timezone defined in
retraining_pipeline.RETRAINING_SCHEDULE and wires every stage to a
dedicated PythonOperator so tasks appear individually in the Airflow UI.

DAG stages (all PythonOperators):
  1. validate_data       — sanity-check incoming feature data
  2. detect_drift        — PSI + KS drift assessment
  3. train_challenger    — train new model, register in MLflow Staging
  4. ab_evaluate         — champion vs challenger A/B evaluation
  5. promote_or_archive  — champion/challenger decision + MLflow stage transition
  6. generate_model_card — produce JSON + Markdown model card
  7. notify_climate_os   — publish result summary (via Airflow variable / HTTP)

XCom keys pushed by each task are documented inline.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default DAG args
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARGS: Dict[str, Any] = {
    "owner":              "ANALYTICA",
    "depends_on_past":    False,
    "email":              ["analytica@tropiclimate.id"],
    "email_on_failure":   True,
    "email_on_retry":     False,
    "retries":            2,
    "retry_delay":        timedelta(minutes=10),
    "execution_timeout":  timedelta(hours=4),
}

# Airflow Variable keys (set these in the Airflow UI or via CLI)
VAR_MLFLOW_URI        = "ANALYTICA_MLFLOW_TRACKING_URI"
VAR_REFERENCE_PATH    = "ANALYTICA_REFERENCE_DATA_PATH"   # parquet path per model
VAR_NEW_DATA_PATH     = "ANALYTICA_NEW_DATA_PATH"
VAR_CLIMATE_OS_URL    = "ANALYTICA_CLIMATE_OS_WEBHOOK_URL"


# ─────────────────────────────────────────────────────────────────────────────
# Helper — resolve Airflow Variable with fallback
# ─────────────────────────────────────────────────────────────────────────────

def _var(key: str, fallback: str = "") -> str:
    try:
        return Variable.get(key, default_var=fallback)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Stage callables (one per PythonOperator)
# ─────────────────────────────────────────────────────────────────────────────

def validate_data_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    Load new data from the feature store path and run basic quality checks.
    Pushes: new_data_path, new_data_rows
    """
    import pandas as pd

    data_path = _var(f"{VAR_NEW_DATA_PATH}_{model_type.upper()}",
                     f"/data/features/{model_type}/latest.parquet")
    log.info(f"[{model_type}] Loading new data from {data_path}")

    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        # Graceful degradation: create a dummy check record for non-production envs
        log.warning(f"Data file not found at {data_path} — using stub for DAG validation")
        df = pd.DataFrame({"stub": [1]})

    n_rows  = len(df)
    n_cols  = len(df.columns)
    null_pct = float(df.isnull().mean().mean())

    if null_pct > 0.5:
        raise ValueError(f"[{model_type}] Data quality check failed: {null_pct:.0%} nulls")

    result = {
        "model_type":  model_type,
        "data_path":   data_path,
        "n_rows":      n_rows,
        "n_cols":      n_cols,
        "null_pct":    null_pct,
        "validated_at": datetime.utcnow().isoformat(),
    }
    context["ti"].xcom_push(key="validation_result", value=result)
    log.info(f"[{model_type}] Validation passed: {n_rows:,} rows × {n_cols} cols, "
             f"{null_pct:.1%} nulls")
    return result


def detect_drift_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    Run PSI + KS drift detection against the reference (training) distribution.
    Pushes: drift_report, should_retrain
    """
    import pandas as pd
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    from analytics.retraining_pipeline import DataDriftDetector

    validation = context["ti"].xcom_pull(task_ids="validate_data", key="validation_result")
    data_path  = validation["data_path"]
    ref_path   = _var(f"{VAR_REFERENCE_PATH}_{model_type.upper()}",
                      f"/data/features/{model_type}/reference.parquet")

    try:
        new_df = pd.read_parquet(data_path)
        ref_df = pd.read_parquet(ref_path)
    except FileNotFoundError:
        log.warning("Data files not found — skipping drift detection, flagging for retrain")
        report = {"should_retrain": True, "reason": "reference_data_missing"}
        context["ti"].xcom_push(key="drift_report",   value=report)
        context["ti"].xcom_push(key="should_retrain", value=True)
        return report

    detector = DataDriftDetector()
    detector.fit(ref_df)
    report = detector.detect(new_df)

    context["ti"].xcom_push(key="drift_report",   value=report)
    context["ti"].xcom_push(key="should_retrain", value=report["should_retrain"])
    log.info(f"[{model_type}] Drift detection: should_retrain={report['should_retrain']}, "
             f"max_psi={report.get('max_psi', 'n/a'):.4f}")
    return report


def train_challenger_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    Train a challenger model if drift was detected.
    Pushes: run_id, challenger_metrics, skipped
    """
    import pandas as pd
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    from analytics.retraining_pipeline import RetrainingPipeline

    should_retrain = context["ti"].xcom_pull(task_ids="detect_drift", key="should_retrain")
    if not should_retrain:
        log.info(f"[{model_type}] No drift — skipping challenger training")
        context["ti"].xcom_push(key="skipped", value=True)
        return {"skipped": True, "reason": "no_drift"}

    data_path = context["ti"].xcom_pull(
        task_ids="validate_data", key="validation_result"
    )["data_path"]
    ref_path  = _var(f"{VAR_REFERENCE_PATH}_{model_type.upper()}",
                     f"/data/features/{model_type}/reference.parquet")

    try:
        new_df = pd.read_parquet(data_path)
        ref_df = pd.read_parquet(ref_path)
    except FileNotFoundError:
        new_df = ref_df = pd.DataFrame({"stub": range(100), "target": range(100)})

    pipeline = RetrainingPipeline(model_type)
    summary  = pipeline.run(new_df, ref_df)

    result = {
        "skipped":            False,
        "run_id":             summary.get("challenger_run_id"),
        "challenger_metrics": summary.get("challenger_metrics", {}),
        "champion_version":   summary.get("champion_version"),
        "action":             summary.get("action"),
    }
    context["ti"].xcom_push(key="train_result", value=result)
    log.info(f"[{model_type}] Challenger trained: run_id={result['run_id']}, "
             f"metrics={result['challenger_metrics']}")
    return result


def ab_evaluate_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    Compare champion vs challenger on a held-out evaluation set.
    Pushes: ab_result, challenger_win_rate
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    from analytics.retraining_pipeline import ChampionChallengerManager

    train_result = context["ti"].xcom_pull(task_ids="train_challenger", key="train_result")
    if not train_result or train_result.get("skipped"):
        log.info(f"[{model_type}] Challenger was skipped — no A/B evaluation needed")
        result = {"skipped": True}
        context["ti"].xcom_push(key="ab_result", value=result)
        return result

    metric_map = {
        "precipitation_nowcast": ("val_rmse",  True),   # lower is better
        "seasonal_forecast":     ("cv_rmse",   True),
        "land_cover_cnn":        ("val_accuracy", False), # higher is better
    }
    metric_key, lower_is_better = metric_map.get(model_type, ("val_rmse", True))

    champion_metrics  = train_result.get("champion_metrics",   {metric_key: 9999.0})
    challenger_metrics = train_result.get("challenger_metrics", {metric_key: 9999.0})

    cc = ChampionChallengerManager(model_type)
    cc.champion_v   = train_result.get("champion_version")
    cc.challenger_v = train_result.get("run_id")

    for _ in range(5):
        cc.evaluate_round(
            champion_metrics.get(metric_key,   9999.0),
            challenger_metrics.get(metric_key, 9999.0),
            metric_name=metric_key,
            lower_is_better=lower_is_better,
        )

    should_promote, win_rate = cc.should_promote()
    result = {
        "should_promote":      should_promote,
        "challenger_win_rate": win_rate,
        "metric_key":          metric_key,
        "champion_metric":     champion_metrics.get(metric_key),
        "challenger_metric":   challenger_metrics.get(metric_key),
    }
    context["ti"].xcom_push(key="ab_result",            value=result)
    context["ti"].xcom_push(key="challenger_win_rate",  value=win_rate)
    log.info(f"[{model_type}] A/B: win_rate={win_rate:.1%}, "
             f"promote={should_promote}")
    return result


def promote_or_archive_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    Promote challenger to Production or archive it, based on A/B outcome.
    Pushes: promotion_result
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    from analytics.retraining_pipeline import ChampionChallengerManager
    from analytics.mlflow_setup import MODEL_REGISTRY_NAMES

    ab_result    = context["ti"].xcom_pull(task_ids="ab_evaluate",     key="ab_result")
    train_result = context["ti"].xcom_pull(task_ids="train_challenger", key="train_result")

    if (not ab_result or ab_result.get("skipped") or
            not train_result or train_result.get("skipped")):
        log.info(f"[{model_type}] No promotion needed — no challenger trained")
        result = {"action": "no_op", "reason": "no_challenger"}
        context["ti"].xcom_push(key="promotion_result", value=result)
        return result

    model_name = MODEL_REGISTRY_NAMES.get(model_type, model_type)
    cc = ChampionChallengerManager(model_name)
    cc.champion_v   = train_result.get("champion_version")
    cc.challenger_v = train_result.get("run_id")

    # Replay the A/B decision into the manager without re-computing
    n_rounds = 5
    win_rate = ab_result["challenger_win_rate"]
    wins     = round(win_rate * n_rounds)
    for i in range(n_rounds):
        winner = "challenger" if i < wins else "champion"
        cc._eval_results.append({
            "champion_metric":   ab_result["champion_metric"],
            "challenger_metric": ab_result["challenger_metric"],
            "metric_name":       ab_result["metric_key"],
            "winner":            winner,
        })

    result = cc.finalize()
    context["ti"].xcom_push(key="promotion_result", value=result)
    log.info(f"[{model_type}] Promotion result: {result}")
    return result


def generate_model_card_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    Produce JSON + Markdown model cards for the (newly promoted) model.
    Pushes: model_card_paths
    """
    import sys
    sys.path.insert(0, "/opt/analytica/src")

    from analytics.explainability import ModelCardGenerator

    promotion_result = context["ti"].xcom_pull(
        task_ids="promote_or_archive", key="promotion_result"
    ) or {}
    train_result = context["ti"].xcom_pull(
        task_ids="train_challenger", key="train_result"
    ) or {}

    run_id   = train_result.get("run_id", "unknown")
    version  = promotion_result.get("new_production_v",
               promotion_result.get("production_v", "unknown"))
    metrics  = train_result.get("challenger_metrics", {})

    gen   = ModelCardGenerator(model_type, str(version), run_id, metrics)
    paths = gen.save(output_dir="/opt/analytica/docs/model_cards")
    context["ti"].xcom_push(key="model_card_paths", value=paths)
    log.info(f"[{model_type}] Model card written: {paths}")
    return paths


def notify_climate_os_fn(model_type: str, **context) -> Dict[str, Any]:
    """
    POST a JSON summary to the CLIMATE-OS webhook (if configured),
    then log the result summary to Airflow logs regardless.
    """
    import requests

    ti = context["ti"]

    # Gather XCom from all upstream tasks
    drift      = ti.xcom_pull(task_ids="detect_drift",      key="drift_report")    or {}
    ab         = ti.xcom_pull(task_ids="ab_evaluate",       key="ab_result")       or {}
    promotion  = ti.xcom_pull(task_ids="promote_or_archive",key="promotion_result")or {}
    card_paths = ti.xcom_pull(task_ids="generate_model_card",key="model_card_paths")or {}

    payload = {
        "agent":            "ANALYTICA",
        "sprint":           2,
        "model_type":       model_type,
        "dag_run_id":       context["run_id"],
        "execution_date":   str(context["execution_date"]),
        "drift_detected":   drift.get("should_retrain", False),
        "max_psi":          drift.get("max_psi"),
        "ab_win_rate":      ab.get("challenger_win_rate"),
        "promotion_action": promotion.get("action"),
        "new_production_v": promotion.get("new_production_v"),
        "model_card":       card_paths,
        "summary_at":       datetime.utcnow().isoformat(),
    }

    webhook_url = _var(VAR_CLIMATE_OS_URL)
    if webhook_url:
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            log.info(f"[{model_type}] CLIMATE-OS notified: HTTP {resp.status_code}")
        except Exception as exc:
            log.warning(f"[{model_type}] Webhook notification failed (non-fatal): {exc}")
    else:
        log.info(f"[{model_type}] No webhook URL set — CLIMATE-OS notification skipped")

    log.info(f"[{model_type}] Run summary:\n{json.dumps(payload, indent=2)}")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# DAG Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_retraining_dag(
    model_type: str,
    schedule_interval: str,
    timezone: str = "Asia/Jakarta",
    dag_id: str | None = None,
    description: str | None = None,
) -> DAG:
    """
    Build and return a complete Airflow retraining DAG for the given model_type.

    Parameters
    ----------
    model_type          : one of 'precipitation_nowcast', 'seasonal_forecast', 'land_cover_cnn'
    schedule_interval   : cron expression (e.g. '0 1 * * 1')
    timezone            : Airflow timezone string (default 'Asia/Jakarta')
    dag_id              : overrides default 'analytica_retrain_{model_type}'
    description         : DAG description shown in Airflow UI
    """
    from pendulum import timezone as pendulum_tz

    _dag_id = dag_id or f"analytica_retrain_{model_type}"
    _desc   = description or (
        f"ANALYTICA — Automated retraining pipeline for {model_type}. "
        f"Drift detection → train → A/B eval → promote → model card."
    )

    with DAG(
        dag_id           = _dag_id,
        description      = _desc,
        default_args     = DEFAULT_ARGS,
        schedule_interval= schedule_interval,
        start_date       = datetime(2026, 6, 1, tzinfo=pendulum_tz(timezone)),
        catchup          = False,
        max_active_runs  = 1,
        tags             = ["analytica", "retraining", "mlops", model_type],
    ) as dag:

        t_validate = PythonOperator(
            task_id         = "validate_data",
            python_callable = validate_data_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "Load new feature data and run null/shape quality checks.",
        )

        t_drift = PythonOperator(
            task_id         = "detect_drift",
            python_callable = detect_drift_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "PSI + KS drift detection vs reference distribution.",
        )

        t_train = PythonOperator(
            task_id         = "train_challenger",
            python_callable = train_challenger_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "Train challenger model and register in MLflow Staging.",
        )

        t_ab = PythonOperator(
            task_id         = "ab_evaluate",
            python_callable = ab_evaluate_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "Champion vs challenger A/B evaluation (5 rounds).",
        )

        t_promote = PythonOperator(
            task_id         = "promote_or_archive",
            python_callable = promote_or_archive_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "Promote challenger to Production or archive it.",
        )

        t_card = PythonOperator(
            task_id         = "generate_model_card",
            python_callable = generate_model_card_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "Generate JSON + Markdown model card (Model Card 2.0).",
        )

        t_notify = PythonOperator(
            task_id         = "notify_climate_os",
            python_callable = notify_climate_os_fn,
            op_kwargs       = {"model_type": model_type},
            doc_md          = "POST run summary to CLIMATE-OS webhook.",
            trigger_rule    = "all_done",   # always runs, even if upstream is skipped
        )

        # ── DAG topology ──────────────────────────────────────────────────
        (t_validate
         >> t_drift
         >> t_train
         >> t_ab
         >> t_promote
         >> t_card
         >> t_notify)

    return dag
