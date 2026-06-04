"""
Model Monitoring DAG — ANALYTICA Sprint 7 L2
dag_id: analytica_model_monitoring
Schedule: 0 4 * * * Asia/Jakarta (daily 04:00 WIB — before A/B testing at 06:00)
SLA: 25 minutes

Pipeline:
  fetch_reference_data
    -> fetch_current_data
    -> check_all_models   (TaskGroup: 4 parallel, one per model)
    -> aggregate_monitoring_summary
    -> emit_monitoring_metrics
    -> trigger_retraining_if_critical
    -> write_monitoring_report

Cross-agent handoff:
  workspace/output/monitoring/monitoring_summary_{YYYYMMDD}.md → VISUALIA
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

DAG_ID      = "analytica_model_monitoring"
SCHEDULE    = "0 4 * * *"
TIMEZONE    = "Asia/Jakarta"
SLA_SECONDS = 25 * 60

REFERENCE_WINDOW_DAYS = 90
CURRENT_WINDOW_DAYS   = 7

MONITORED_MODELS = [
    "xgb_precipitation",
    "prophet_seasonal",
    "cnn_landcover",
    "lstm_streamflow",
]

OUTPUT_DIR = Path("workspace/output/monitoring")


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def fetch_reference_data(**context) -> None:
    """Load last-90-day feature distributions from FeatureStoreClient (baseline window)."""
    from src.data.feature_store_client import FeatureStoreClient

    run_ds = context["ds"]
    end_dt = datetime.strptime(run_ds, "%Y-%m-%d")
    end    = end_dt.date()
    start  = (end_dt - timedelta(days=REFERENCE_WINDOW_DAYS)).date()

    fs = FeatureStoreClient()
    dfs = {}
    for entity_type in ["station_id", "grid_cell_id", "watershed_id"]:
        try:
            df = fs.get_historical_features(
                start_date=start, end_date=end, entity_types=[entity_type]
            )
            data_path = OUTPUT_DIR / f"ref_{entity_type}_{run_ds.replace('-','')}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(str(data_path), index=False)
            dfs[entity_type] = str(data_path)
            logger.info("Reference data [%s]: %d rows for %s→%s", entity_type, len(df), start, end)
        except Exception as exc:
            logger.warning("Could not fetch reference data for %s: %s", entity_type, exc)
            dfs[entity_type] = None

    context["ti"].xcom_push(key="ref_paths", value=dfs)


def fetch_current_data(**context) -> None:
    """Load last-7-day features (current window for drift comparison)."""
    from src.data.feature_store_client import FeatureStoreClient

    run_ds = context["ds"]
    end_dt = datetime.strptime(run_ds, "%Y-%m-%d")
    end    = end_dt.date()
    start  = (end_dt - timedelta(days=CURRENT_WINDOW_DAYS)).date()

    fs = FeatureStoreClient()
    dfs = {}
    for entity_type in ["station_id", "grid_cell_id", "watershed_id"]:
        try:
            df = fs.get_historical_features(
                start_date=start, end_date=end, entity_types=[entity_type]
            )
            data_path = OUTPUT_DIR / f"cur_{entity_type}_{run_ds.replace('-','')}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(str(data_path), index=False)
            dfs[entity_type] = str(data_path)
            logger.info("Current data [%s]: %d rows for %s→%s", entity_type, len(df), start, end)
        except Exception as exc:
            logger.warning("Could not fetch current data for %s: %s", entity_type, exc)
            dfs[entity_type] = None

    context["ti"].xcom_push(key="cur_paths", value=dfs)


def _entity_type_for_model(model_id: str) -> str:
    mapping = {
        "xgb_precipitation":  "station_id",
        "xgb_precip_nowcast": "station_id",
        "prophet_seasonal":   "watershed_id",
        "cnn_landcover":      "grid_cell_id",
        "lstm_streamflow":    "watershed_id",
    }
    return mapping.get(model_id, "station_id")


def _make_check_model(model_id: str):
    """Factory to create a model-specific check callable for the TaskGroup."""
    def check_model(**context):
        import pandas as pd
        from src.monitoring.model_monitor import ModelMonitor

        ti        = context["ti"]
        run_ds    = context["ds"]
        run_date  = datetime.strptime(run_ds, "%Y-%m-%d").date()
        ref_paths = ti.xcom_pull(key="ref_paths", task_ids="fetch_reference_data") or {}
        cur_paths = ti.xcom_pull(key="cur_paths", task_ids="fetch_current_data")   or {}

        entity_type = _entity_type_for_model(model_id)
        ref_path    = ref_paths.get(entity_type)
        cur_path    = cur_paths.get(entity_type)

        monitor = ModelMonitor()

        ref_df: Optional[pd.DataFrame] = None
        cur_df: Optional[pd.DataFrame] = None
        if ref_path and Path(ref_path).exists():
            ref_df = pd.read_parquet(ref_path)
        if cur_path and Path(cur_path).exists():
            cur_df = pd.read_parquet(cur_path)

        report = monitor.run_full_check(
            model_id=model_id,
            reference_df=ref_df,
            current_df=cur_df,
            run_date=run_date,
        )

        ti.xcom_push(key=f"report_{model_id}", value=report.to_dict())
        logger.info("Model check [%s] overall_status=%s retrain=%s",
                    model_id, report.overall_status, report.retrain_triggered)

    check_model.__name__ = f"check_{model_id}"
    return check_model


def aggregate_monitoring_summary(**context) -> None:
    """Count HEALTHY/DEGRADED/CRITICAL per model; identify worst drift features."""
    ti = context["ti"]
    summary = {"models": {}, "totals": {"ok": 0, "warning": 0, "degraded": 0, "critical": 0}}

    for model_id in MONITORED_MODELS:
        report = ti.xcom_pull(key=f"report_{model_id}", task_ids=f"check_all_models.check_{model_id}")
        if not report:
            summary["models"][model_id] = {"status": "unknown"}
            continue
        status = report.get("overall_status", "unknown")
        drift  = report.get("drift") or {}
        top_drifted = (drift.get("drifted_features") or [])[:3]
        summary["models"][model_id] = {
            "status":            status,
            "retrain_triggered": report.get("retrain_triggered", False),
            "top_drifted":       top_drifted,
        }
        if status in summary["totals"]:
            summary["totals"][status] += 1

    ti.xcom_push(key="monitoring_summary", value=summary)
    logger.info("Monitoring aggregate: %s", summary["totals"])


def emit_monitoring_metrics(**context) -> None:
    """Emit MODEL_DRIFT_SCORE Prometheus gauges per model + feature."""
    ti      = context["ti"]
    summary = ti.xcom_pull(key="monitoring_summary", task_ids="aggregate_monitoring_summary") or {}

    try:
        from src.data.metrics import MODEL_DRIFT_SCORE
    except ImportError:
        logger.warning("Prometheus metrics unavailable — skipping emit_monitoring_metrics")
        return

    for model_id in MONITORED_MODELS:
        report = ti.xcom_pull(key=f"report_{model_id}", task_ids=f"check_all_models.check_{model_id}")
        if not report:
            continue
        drift = report.get("drift") or {}
        for feat_result in (drift.get("feature_results") or []):
            col   = feat_result.get("feature", "unknown")
            psi   = feat_result.get("psi")
            ks_p  = feat_result.get("ks_pvalue")
            js_d  = feat_result.get("js_divergence")
            if psi  is not None:
                MODEL_DRIFT_SCORE.labels(model_id=model_id, feature=col, drift_type="psi").set(psi)
            if ks_p is not None:
                MODEL_DRIFT_SCORE.labels(model_id=model_id, feature=col, drift_type="ks").set(ks_p)
            if js_d is not None:
                MODEL_DRIFT_SCORE.labels(model_id=model_id, feature=col, drift_type="js").set(js_d)


def trigger_retraining_if_critical(**context) -> None:
    """Set RETRAIN_{MODEL_ID}=true for any CRITICAL model (belt-and-suspenders — also done in ModelMonitor)."""
    from airflow.models import Variable

    VARIABLE_MAP = {
        "xgb_precipitation": "RETRAIN_XGB",
        "prophet_seasonal":  "RETRAIN_PROPHET",
        "cnn_landcover":     "RETRAIN_CNN",
        "lstm_streamflow":   "RETRAIN_LSTM",
    }
    ti = context["ti"]
    triggered = []
    for model_id in MONITORED_MODELS:
        report = ti.xcom_pull(key=f"report_{model_id}", task_ids=f"check_all_models.check_{model_id}")
        if report and report.get("overall_status") == "critical":
            var = VARIABLE_MAP.get(model_id)
            if var:
                Variable.set(var, "true")
                triggered.append(model_id)
                logger.warning("CRITICAL model %s — set %s=true", model_id, var)

    context["ti"].xcom_push(key="triggered_retraining", value=triggered)
    if triggered:
        logger.warning("Retraining triggered for: %s", triggered)
    else:
        logger.info("No models require immediate retraining")


def write_monitoring_report(**context) -> None:
    """
    Write monitoring_summary_{YYYYMMDD}.md for cross-agent handoff → VISUALIA.
    """
    from datetime import datetime as dt
    ti      = context["ti"]
    run_ds  = context["ds"]
    summary = ti.xcom_pull(key="monitoring_summary",     task_ids="aggregate_monitoring_summary") or {}
    triggered = ti.xcom_pull(key="triggered_retraining", task_ids="trigger_retraining_if_critical") or []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"monitoring_summary_{run_ds.replace('-', '')}.md"

    totals = summary.get("totals", {})
    models = summary.get("models", {})

    status_emoji = {"ok": "✅", "warning": "⚠️", "degraded": "🟠", "critical": "🔴", "unknown": "❓"}

    rows = ""
    for mid, mdata in models.items():
        st  = mdata.get("status", "unknown")
        emoji = status_emoji.get(st, "❓")
        drifted = ", ".join(mdata.get("top_drifted", [])) or "—"
        retrain = "✅ Triggered" if mdata.get("retrain_triggered") else "—"
        rows += f"| `{mid}` | {emoji} {st.upper()} | {drifted} | {retrain} |\n"

    retraining_section = ""
    if triggered:
        retraining_section = f"\n## ⚡ Retraining Triggered\n\n" + \
                             "\n".join(f"- `{m}` — RETRAIN variable set" for m in triggered) + "\n"

    md = f"""# ANALYTICA Model Monitoring Summary — {run_ds}

> Generated: {dt.now(tz=None).strftime('%Y-%m-%d %H:%M WIB')} | DAG: `{DAG_ID}`

## Summary

| Status | Count |
|--------|-------|
| ✅ OK | {totals.get('ok', 0)} |
| ⚠️ Warning | {totals.get('warning', 0)} |
| 🟠 Degraded | {totals.get('degraded', 0)} |
| 🔴 Critical | {totals.get('critical', 0)} |

## Model Status

| Model | Status | Top Drifted Features | Retraining |
|-------|--------|----------------------|------------|
{rows}
{retraining_section}
## Next Actions

- DEGRADED models: monitor for 2 additional days before triggering retraining
- CRITICAL models: retraining DAG auto-triggered via Airflow Variable gate
- VISUALIA: render drift heatmap from `workspace/output/monitoring/drift_{{model_id}}_*.json`
- API-GATEWAY: ensemble confidence intervals updated in `workspace/output/ensemble/latest_ensemble.json`

---
*Cross-agent: VISUALIA reads this file for dashboard update. COMPLIANCE reviews for PP 71/2019 audit trail.*
"""

    out_path.write_text(md)
    logger.info("Monitoring report written: %s", out_path)

    # Also write a JSON index for API-GATEWAY
    index_path = OUTPUT_DIR / f"monitoring_index_{run_ds.replace('-', '')}.json"
    import json as _json
    index_path.write_text(_json.dumps({
        "run_date":         run_ds,
        "summary":          summary,
        "triggered_models": triggered,
        "report_md":        str(out_path),
        "generated_at":     dt.now(tz=None).isoformat(),
    }, indent=2))


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

default_args = {
    "owner":            "analytica",
    "depends_on_past":  False,
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "sla":              timedelta(seconds=SLA_SECONDS),
}

with DAG(
    dag_id=DAG_ID,
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["analytica", "monitoring", "drift"],
    description="Daily model monitoring: data drift (PSI/KS/JS) + performance degradation; triggers retraining DAGs",
) as dag:

    t_ref  = PythonOperator(task_id="fetch_reference_data", python_callable=fetch_reference_data)
    t_cur  = PythonOperator(task_id="fetch_current_data",   python_callable=fetch_current_data)

    with TaskGroup("check_all_models") as tg_check:
        check_tasks = [
            PythonOperator(
                task_id=f"check_{model_id}",
                python_callable=_make_check_model(model_id),
            )
            for model_id in MONITORED_MODELS
        ]

    t_agg     = PythonOperator(task_id="aggregate_monitoring_summary",  python_callable=aggregate_monitoring_summary)
    t_emit    = PythonOperator(task_id="emit_monitoring_metrics",        python_callable=emit_monitoring_metrics)
    t_trigger = PythonOperator(task_id="trigger_retraining_if_critical", python_callable=trigger_retraining_if_critical)
    t_report  = PythonOperator(task_id="write_monitoring_report",        python_callable=write_monitoring_report)

    [t_ref, t_cur] >> tg_check >> t_agg >> t_emit >> t_trigger >> t_report
