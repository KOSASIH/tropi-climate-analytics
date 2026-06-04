"""
Airflow DAG — Flood Early Warning (30-minute)
HYDROLOGIS Sprint 4 | Deliverable 2

dag_id:   flood_early_warning_30min
schedule: */30 * * * * (Asia/Jakarta)
SLA:      10 minutes

Task graph:
  fetch_qpe → fetch_gauge_readings → run_streamflow_forecast → run_flood_inundation → emit_metrics
                                                                        │
                                                              EMERGENCY stage detected
                                                                        ↓
                                                    TriggerDagRunOperator → emergency_flood_alert_dag
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("WORKSPACE_ROOT", "workspace")

# River mean_annual_q defaults (fallback when gauge data unavailable)
RIVER_MEAN_ANNUAL_Q = {
    "ciliwung": 38.0,
    "brantas":  230.0,
    "solo":     310.0,
}

default_args = {
    "owner":             "hydrologis",
    "depends_on_past":   False,
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
}

# ---------------------------------------------------------------------------
# Task 1: fetch_qpe
# ---------------------------------------------------------------------------

def fetch_qpe(**context) -> dict:
    """
    Read latest QPE fusion output from workspace/output/qpe/.
    Returns the most-recently modified QPE JSON, or an empty dict on miss.
    """
    import glob

    qpe_dir = os.path.join(WORKSPACE, "output", "qpe")
    os.makedirs(qpe_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(qpe_dir, "*.json")), key=os.path.getmtime, reverse=True)
    if not files:
        logger.warning("No QPE output found in %s — using zero precipitation fallback", qpe_dir)
        return {"precip_mm_6h": 0.0, "source": "fallback", "path": None}

    with open(files[0]) as fh:
        data = json.load(fh)

    precip = float(data.get("precip_mm_6h", data.get("mean_precip_mm", 0.0)))
    logger.info("QPE fetched: %.1f mm/6h from %s", precip, files[0])
    return {"precip_mm_6h": precip, "source": files[0], "path": files[0]}


# ---------------------------------------------------------------------------
# Task 2: fetch_gauge_readings
# ---------------------------------------------------------------------------

def fetch_gauge_readings(**context) -> dict:
    """
    Fetch upstream gauge readings from BMKG HIMET API.

    Production: POST to BMKG HIMET REST endpoint with bearer token.
    Sprint 4:
      - Read from workspace/data/bmkg_gauge_latest.json if exists
      - Else use mean_annual_q defaults from RIVER_MEAN_ANNUAL_Q
    """
    gauge_path = os.path.join(WORKSPACE, "data", "bmkg_gauge_latest.json")

    if os.path.exists(gauge_path):
        try:
            with open(gauge_path) as fh:
                gauges = json.load(fh)
            logger.info("BMKG gauge data loaded from %s", gauge_path)
            return {"current_q": gauges, "source": "bmkg_file"}
        except Exception as exc:
            logger.warning("BMKG gauge file read failed (%s) — using defaults", exc)

    # Fallback: mean annual discharge per river
    logger.info("Using mean_annual_q defaults for gauge readings")
    return {"current_q": RIVER_MEAN_ANNUAL_Q.copy(), "source": "defaults"}


# ---------------------------------------------------------------------------
# Task 3: run_streamflow_forecast
# ---------------------------------------------------------------------------

def run_streamflow_forecast(**context) -> dict:
    """
    Run StreamflowForecastEngine ensemble (LSTM + HBV) for all 3 rivers.
    Returns summary including rivers in WARNING/EMERGENCY stage.
    """
    from src.hydrology.streamflow_forecast import StreamflowForecastEngine

    ti = context["ti"]
    qpe_data   = ti.xcom_pull(task_ids="fetch_qpe")
    gauge_data = ti.xcom_pull(task_ids="fetch_gauge_readings")

    precip_mm_6h = qpe_data.get("precip_mm_6h", 0.0)
    current_q    = gauge_data.get("current_q", RIVER_MEAN_ANNUAL_Q)

    engine = StreamflowForecastEngine()
    status = engine.run(
        precip_mm_6h=precip_mm_6h,
        current_q=current_q,
    )

    has_emergency = len(status.rivers_in_emergency) > 0
    logger.info(
        "Streamflow forecast complete | warning=%s emergency=%s",
        status.rivers_in_warning,
        status.rivers_in_emergency,
    )

    # Build peak discharge map per river (24h horizon)
    peak_q_map: dict[str, float] = {}
    for f in status.forecasts:
        if f.forecast_horizon_hours == 24:
            peak_q_map[f.river_id] = f.peak_discharge_m3s

    return {
        "rivers_in_warning":   status.rivers_in_warning,
        "rivers_in_emergency": status.rivers_in_emergency,
        "has_emergency":       has_emergency,
        "peak_q_24h":          peak_q_map,
        "output_paths":        status.output_paths,
    }


# ---------------------------------------------------------------------------
# Task 4: run_flood_inundation
# ---------------------------------------------------------------------------

def run_flood_inundation(**context) -> dict:
    """
    Run FloodInundationMapper for rivers in WARNING or EMERGENCY stage only.
    In-bank rivers (NORMAL/WATCH) are skipped to reduce compute.
    """
    from datetime import date

    from src.hydrology.flood_inundation import FloodInundationMapper

    ti = context["ti"]
    sf_result = ti.xcom_pull(task_ids="run_streamflow_forecast")

    warning_rivers   = sf_result.get("rivers_in_warning", [])
    emergency_rivers = sf_result.get("rivers_in_emergency", [])
    active_rivers    = list(set(warning_rivers + emergency_rivers))

    if not active_rivers:
        logger.info("No rivers in WARNING/EMERGENCY — skipping flood inundation run")
        return {"skipped": True, "reason": "no rivers above WARNING threshold"}

    peak_q = sf_result.get("peak_q_24h", {})
    forecasts = {
        river_id: {6: peak_q.get(river_id, 0.0) * 0.7,
                   12: peak_q.get(river_id, 0.0) * 0.9,
                   24: peak_q.get(river_id, 0.0)}
        for river_id in active_rivers
        if river_id in peak_q
    }

    mapper = FloodInundationMapper()
    status = mapper.run(forecasts=forecasts, reference_date=date.today())

    logger.info(
        "Flood inundation mapped | rivers=%s total_km2=%.1f",
        active_rivers,
        status.total_inundated_km2,
    )

    return {
        "skipped":            False,
        "rivers_mapped":      active_rivers,
        "total_inundated_km2": status.total_inundated_km2,
        "results_count":      len(status.results),
    }


# ---------------------------------------------------------------------------
# Task 5: emit_metrics
# ---------------------------------------------------------------------------

def emit_metrics(**context) -> dict:
    """
    Confirm tropi_pipeline_last_ingestion_success_timestamp_seconds is updated.
    Also push final Prometheus batch.
    """
    from src.hydrology.metrics import push_metrics, record_ingestion_success

    record_ingestion_success("flood_early_warning_30min")
    push_metrics()

    ti = context["ti"]
    sf_result = ti.xcom_pull(task_ids="run_streamflow_forecast")

    logger.info(
        "Metrics emitted | pipeline=flood_early_warning_30min warning=%s emergency=%s",
        sf_result.get("rivers_in_warning"),
        sf_result.get("rivers_in_emergency"),
    )
    return {"metrics_pushed": True}


# ---------------------------------------------------------------------------
# Branch: check for EMERGENCY stage
# ---------------------------------------------------------------------------

def check_emergency_stage(**context) -> str:
    ti = context["ti"]
    sf_result = ti.xcom_pull(task_ids="run_streamflow_forecast")
    if sf_result.get("has_emergency"):
        logger.critical(
            "EMERGENCY flood stage detected: %s — triggering emergency_flood_alert_dag",
            sf_result.get("rivers_in_emergency"),
        )
        return "trigger_emergency_alert"
    return "no_emergency"


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="flood_early_warning_30min",
    description="30-minute flood early warning pipeline: QPE → gauge → LSTM+HBV forecast → inundation → metrics",
    schedule="*/30 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["hydrologis", "flood", "streamflow", "realtime"],
    doc_md="""
## HYDROLOGIS — Flood Early Warning (30-minute)

**Schedule:** Every 30 minutes (Asia/Jakarta)  
**SLA:** 10 minutes  
**Rivers:** Ciliwung (DKI Jakarta) · Brantas (Jawa Timur) · Solo (Jawa Tengah)  

**Pipeline:**  
`fetch_qpe` → `fetch_gauge_readings` → `run_streamflow_forecast` → `run_flood_inundation` → `emit_metrics`  

**EMERGENCY routing:** Any river hitting EMERGENCY stage triggers `emergency_flood_alert_dag`  
**Prometheus:** `tropi_pipeline_last_ingestion_success_timestamp_seconds{pipeline=flood_early_warning_30min}`
    """,
    dagrun_timeout=timedelta(minutes=10),
) as dag:

    t_qpe = PythonOperator(
        task_id="fetch_qpe",
        python_callable=fetch_qpe,
    )

    t_gauge = PythonOperator(
        task_id="fetch_gauge_readings",
        python_callable=fetch_gauge_readings,
    )

    t_forecast = PythonOperator(
        task_id="run_streamflow_forecast",
        python_callable=run_streamflow_forecast,
    )

    t_inundation = PythonOperator(
        task_id="run_flood_inundation",
        python_callable=run_flood_inundation,
    )

    t_metrics = PythonOperator(
        task_id="emit_metrics",
        python_callable=emit_metrics,
    )

    t_branch = BranchPythonOperator(
        task_id="check_emergency_stage",
        python_callable=check_emergency_stage,
    )

    t_emergency = TriggerDagRunOperator(
        task_id="trigger_emergency_alert",
        trigger_dag_id="emergency_flood_alert_dag",   # stub — see dags/emergency_flood_alert_dag.py
        conf={"alert_type": "EMERGENCY_FLOOD"},
        wait_for_completion=False,
    )

    t_no_emergency = EmptyOperator(task_id="no_emergency")

    # Task dependencies
    [t_qpe, t_gauge] >> t_forecast >> t_inundation >> t_metrics >> t_branch >> [t_emergency, t_no_emergency]
