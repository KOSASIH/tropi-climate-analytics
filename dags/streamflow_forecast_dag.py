"""
streamflow_forecast_dag.py — Sprint 6 E2
dag_id: hydrologis_streamflow_forecast
schedule: */30 Asia/Jakarta (aligned to qpe_ingest_dag)
SLA: 8 minutes

Tasks:
  load_inputs           — validate QPE age < 35min; skip if stale
  run_streamflow_forecast — StreamflowForecaster.run() for all 3 rivers
  log_forecast_metrics  — push per-horizon Prometheus Gauges
  write_forecast_output — confirm output JSON written; record_ingestion_success

On_success: record_ingestion_success('streamflow_forecast_30min')
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
QPE_PATH  = WORKSPACE / "data" / "qpe_latest_validated.json"
RIVERS    = ["ciliwung", "brantas", "solo"]
HORIZONS  = [6, 12, 24]

_default_args = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def load_inputs(**context) -> dict:
    """
    Validate QPE file exists and is fresh (age < 35 min).
    Raises AirflowSkipException if QPE is stale — downstream tasks will skip.
    """
    from airflow.exceptions import AirflowSkipException

    if not QPE_PATH.exists():
        raise AirflowSkipException(
            f"QPE file not found at {QPE_PATH}. Skipping streamflow forecast run."
        )

    with open(QPE_PATH) as f:
        qpe = json.load(f)

    # Check timestamp field
    ts_str = qpe.get("validated_at") or qpe.get("timestamp")
    if ts_str:
        try:
            ts  = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - ts
            if age > timedelta(minutes=35):
                raise AirflowSkipException(
                    f"QPE data is {age.seconds // 60}min old (max 35min). "
                    "Skipping streamflow forecast run."
                )
            logger.info("QPE age check OK: %.1f min", age.seconds / 60)
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse QPE timestamp '%s': %s — proceeding", ts_str, exc)
    else:
        logger.warning("QPE file has no timestamp field; skipping age check")

    logger.info("QPE inputs validated | path=%s coverage=%.1f%%",
                QPE_PATH, qpe.get("coverage_pct", 0.0))
    return {"qpe_valid": True, "qpe_path": str(QPE_PATH)}


def run_streamflow_forecast(**context) -> dict:
    """
    Run StreamflowForecaster.run() for Ciliwung, Brantas, and Solo.
    Pushes per-river StreamflowForecastResult (serialised) via XCom.
    """
    from src.hydrology.streamflow_forecast import StreamflowForecaster

    forecaster = StreamflowForecaster()
    results    = {}
    errors     = []

    for river_id in RIVERS:
        try:
            result = forecaster.run(river_id=river_id, horizons=HORIZONS)
            results[river_id] = {
                "status":       result.status,
                "peak_cms_max": result.peak_cms_max,
                "output_path":  result.output_path,
                "horizons": [
                    {
                        "horizon_hr": h.horizon_hr,
                        "peak_cms":   h.peak_cms,
                        "confidence": h.confidence,
                    }
                    for h in result.horizons
                ],
                "warnings": result.warnings,
            }
            logger.info(
                "Forecast OK | river=%s peak_max=%.1f m³/s status=%s",
                river_id, result.peak_cms_max, result.status,
            )

            # --- Flood threshold evaluation ---
            try:
                from src.hydrology.flood_threshold_evaluator import FloodThresholdEvaluator
                evaluator = FloodThresholdEvaluator()
                eval_r    = evaluator.evaluate(river_id=river_id, forecast=result)
                results[river_id]["flood_stage"]  = eval_r.flood_stage
                results[river_id]["alert_triggered"] = eval_r.trigger_dispatched
                if eval_r.flood_stage != "NONE":
                    logger.warning(
                        "FLOOD ALERT | river=%s stage=%s peak=%.1f m³/s triggered=%s",
                        river_id, eval_r.flood_stage, eval_r.peak_cms, eval_r.trigger_dispatched,
                    )
            except Exception as eval_exc:
                logger.warning("Threshold evaluation error for %s: %s", river_id, eval_exc)
                results[river_id]["flood_stage"]    = "UNKNOWN"
                results[river_id]["alert_triggered"] = False

        except Exception as exc:
            logger.error("Forecast FAILED | river=%s error=%s", river_id, exc, exc_info=True)
            errors.append(f"{river_id}: {exc}")
            results[river_id] = {"status": "failed", "error": str(exc)}

    if errors:
        logger.warning("Streamflow forecast completed with errors: %s", errors)

    context["ti"].xcom_push(key="forecast_results", value=results)
    return results


def log_forecast_metrics(**context) -> dict:
    """
    Re-push per-river, per-horizon Prometheus Gauges from XCom results.
    Also logs summary to Airflow task logs for observability.
    """
    ti      = context["ti"]
    results = ti.xcom_pull(task_ids="run_streamflow_forecast", key="forecast_results")
    if not results:
        logger.warning("No forecast results in XCom; skipping metrics logging")
        return {}

    summary = {}
    for river_id, data in results.items():
        if data.get("status") == "failed":
            logger.warning("Skipping metrics for failed river: %s", river_id)
            continue

        peak_max = data.get("peak_cms_max", 0.0)
        flood_stage = data.get("flood_stage", "NONE")
        logger.info(
            "Metrics | river=%-10s peak_max=%7.1f m³/s flood_stage=%-10s alert=%s",
            river_id, peak_max, flood_stage, data.get("alert_triggered", False),
        )
        summary[river_id] = {"peak_cms_max": peak_max, "flood_stage": flood_stage}

    return summary


def write_forecast_output(**context) -> str:
    """
    Confirm output paths exist; record ingestion success metric.
    Returns a comma-separated list of confirmed output paths.
    """
    ti      = context["ti"]
    results = ti.xcom_pull(task_ids="run_streamflow_forecast", key="forecast_results")
    if not results:
        logger.warning("No forecast results to confirm")
        return "no_outputs"

    confirmed = []
    for river_id, data in results.items():
        out_path = data.get("output_path")
        if out_path and Path(out_path).exists():
            confirmed.append(out_path)
            logger.info("Output confirmed | river=%s path=%s", river_id, out_path)
        else:
            logger.warning("Output NOT found | river=%s path=%s", river_id, out_path)

    # Record ingestion success
    try:
        from src.hydrology.metrics import record_ingestion_success
        record_ingestion_success("streamflow_forecast_30min")
        logger.info("record_ingestion_success('streamflow_forecast_30min') called")
    except ImportError:
        # Fallback: set INGESTION_SUCCESS_TS directly
        try:
            import time
            from src.hydrology.metrics import INGESTION_SUCCESS_TS
            INGESTION_SUCCESS_TS.labels(source="streamflow_forecast_30min").set(time.time())
        except Exception as exc:
            logger.debug("Could not set INGESTION_SUCCESS_TS: %s", exc)

    return ",".join(confirmed)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id             = "hydrologis_streamflow_forecast",
    description        = "Sprint 6: Rational-method streamflow forecast for Ciliwung, Brantas, Solo — every 30 min",
    default_args       = _default_args,
    schedule_interval  = "*/30 * * * *",
    start_date         = days_ago(1),
    catchup            = False,
    max_active_runs    = 1,
    tags               = ["hydrologis", "streamflow", "sprint6"],
    dagrun_timeout     = timedelta(minutes=8),
) as dag:

    t_load = PythonOperator(
        task_id         = "load_inputs",
        python_callable = load_inputs,
    )

    t_forecast = PythonOperator(
        task_id         = "run_streamflow_forecast",
        python_callable = run_streamflow_forecast,
    )

    t_metrics = PythonOperator(
        task_id         = "log_forecast_metrics",
        python_callable = log_forecast_metrics,
    )

    t_output = PythonOperator(
        task_id         = "write_forecast_output",
        python_callable = write_forecast_output,
    )

    t_load >> t_forecast >> t_metrics >> t_output
