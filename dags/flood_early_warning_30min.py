# ============================================================
# DAG: flood_early_warning_30min
# LSTM Streamflow Forecast — Ciliwung / Brantas / Solo
# Schedule: Every 30 minutes
# SLA: Forecast issued within 25 min of QPE availability
# Owner: HYDROLOGIS | Infra: CLOUD-FORGE
# ============================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

RIVERS = ["ciliwung", "brantas", "solo"]

DEFAULT_ARGS = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "email":            ["alerts@tropi-climate.id", "hydrologis@tropi-climate.id"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=3),
    "sla":              timedelta(minutes=25),
    "execution_timeout": timedelta(minutes=28),
}


@dag(
    dag_id="flood_early_warning_30min",
    description="LSTM 6-24h streamflow forecast + flood early warning for Ciliwung, Brantas, Solo rivers",
    schedule_interval="*/30 * * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["hydrology", "flood", "lstm", "early-warning", "30min"],
    doc_md="""
    ## Flood Early Warning DAG
    Runs after `qpe_fusion_30min` to consume QPE output and drive LSTM-based
    streamflow forecasts for three major Indonesian watersheds:
    - **Ciliwung** (DKI Jakarta flood risk)
    - **Brantas** (East Java, Surabaya)
    - **Solo** (Central Java)

    Generates 6h / 12h / 24h lead-time flood stage forecasts and triggers
    BPBD alert messages when thresholds are exceeded.

    **Pipeline class**: `src/hydrology/flood/FloodEarlyWarningPipeline`
    """,
)
def flood_early_warning_dag():

    # Wait for QPE fusion to complete in the same half-hour slot
    wait_for_qpe = ExternalTaskSensor(
        task_id="wait_for_qpe_fusion",
        external_dag_id="qpe_fusion_30min",
        external_task_id="store_qpe_output",
        timeout=1500,            # 25 min max wait
        poke_interval=60,
        mode="reschedule",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
    )

    @task(task_id="load_qpe_and_antecedent_state")
    def load_qpe_and_state(**context) -> dict:
        """Read latest QPE grid and antecedent soil/flow state from PostGIS."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.flood import FloodEarlyWarningPipeline

        execution_dt = context["execution_date"]
        pipeline = FloodEarlyWarningPipeline(
            db_url=Variable.get("DATABASE_URL"),
            redis_url=Variable.get("REDIS_URL"),
        )
        state = pipeline.load_input_state(execution_dt)
        log.info("Loaded QPE + antecedent state for %d watersheds", len(state["watersheds"]))
        return state

    @task.expand_kwargs(
        task_id="lstm_forecast",
        map_kwargs_func=lambda ctx: [{"river": r} for r in RIVERS],
    )
    def lstm_forecast_river(river: str, input_state: dict, **context) -> dict:
        """Run LSTM streamflow forecast for a single watershed (fan-out over 3 rivers)."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.flood import FloodEarlyWarningPipeline

        execution_dt = context["execution_date"]
        pipeline = FloodEarlyWarningPipeline(
            db_url=Variable.get("DATABASE_URL"),
            mlflow_tracking_uri=Variable.get("MLFLOW_TRACKING_URI"),
        )
        forecast = pipeline.run_lstm_forecast(
            river=river,
            input_state=input_state,
            lead_hours=[6, 12, 24],
            execution_dt=execution_dt,
        )
        log.info(
            "[%s] Forecast: 6h=%.2fm 12h=%.2fm 24h=%.2fm (threshold=%.2fm)",
            river.upper(),
            forecast["stage_6h"], forecast["stage_12h"], forecast["stage_24h"],
            forecast["alert_threshold"],
        )
        return forecast

    @task(task_id="evaluate_alert_thresholds")
    def evaluate_alerts(forecasts: list[dict], **context) -> dict:
        """Compare forecasts against BPBD alert thresholds and produce alert payload."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.flood import FloodEarlyWarningPipeline

        pipeline = FloodEarlyWarningPipeline(
            db_url=Variable.get("DATABASE_URL"),
        )
        alert_payload = pipeline.evaluate_thresholds(forecasts)
        active_alerts = [r for r in alert_payload["rivers"] if r["alert_level"] > 0]
        log.info(
            "Alert evaluation: %d active alerts (levels: %s)",
            len(active_alerts),
            {r["river"]: r["alert_level"] for r in active_alerts},
        )
        return alert_payload

    @task(task_id="publish_forecast_and_alerts")
    def publish_results(alert_payload: dict, **context) -> None:
        """Write forecasts to PostGIS, publish Kafka events, trigger BPBD alerts if needed."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.flood import FloodEarlyWarningPipeline

        pipeline = FloodEarlyWarningPipeline(
            db_url=Variable.get("DATABASE_URL"),
            kafka_brokers=Variable.get("KAFKA_BOOTSTRAP_BROKERS"),
            bpbd_webhook_url=Variable.get("BPBD_ALERT_WEBHOOK_URL", default_var=None),
        )
        pipeline.publish(
            alert_payload=alert_payload,
            kafka_topic="flood.early_warning",
        )
        if alert_payload.get("has_active_alerts"):
            log.warning("🚨 FLOOD ALERT ISSUED — %d river(s) above threshold",
                        len([r for r in alert_payload["rivers"] if r["alert_level"] > 0]))

    # ── Task graph ──────────────────────────────────────────
    state     = load_qpe_and_state()
    forecasts = lstm_forecast_river.partial(input_state=state).expand(river=RIVERS)
    alerts    = evaluate_alerts(forecasts)

    wait_for_qpe >> state
    publish_results(alerts)


flood_early_warning_dag_instance = flood_early_warning_dag()
