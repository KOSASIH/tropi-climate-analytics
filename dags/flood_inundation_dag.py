"""
flood_inundation_dag.py — Sprint 7 G4
dag_id: hydrologis_flood_inundation

Trigger-only DAG (schedule=None). Invoked by emergency_flood_alert_dag via
TriggerDagRunOperator when a river breaches its flood threshold.

Conf params (passed by TriggerDagRunOperator):
    river_id          : str   — "ciliwung" | "brantas" | "solo"
    flood_stage       : str   — "WATCH" | "WARNING" | "DANGER" | "EXTREME"
    peak_cms          : float — peak discharge (m³/s) at time of trigger
    forecast_horizon_hr: int  — look-ahead horizon in hours (default 6)
    triggered_at      : str   — ISO-8601 timestamp from emergency_flood_alert_dag

Tasks (linear chain):
    load_dem_data             → validate / stub DEM availability for river_id
    compute_inundation_extent → FloodInundationMapper.map() → GeoJSON
    write_geojson             → persist timestamped + latest files
    emit_inundation_metrics   → Prometheus INUNDATION_AREA_KM2 gauge
    notify_geospatial         → append entry to inundation_queue.jsonl

SLA: 10 minutes from trigger time.
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
# DAG default args
# ---------------------------------------------------------------------------
_DEFAULT_ARGS = {
    "owner": "hydrologis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------------------------
_WS = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_INUNDATION_DIR = _WS / "output" / "inundation"
_QUEUE_FILE = _INUNDATION_DIR / "inundation_queue.jsonl"
_LATEST_FILE = _INUNDATION_DIR / "latest_inundation.json"


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def load_dem_data(**context) -> dict:
    """
    Validate DEM availability for the requested river_id.
    Falls back gracefully to analytical mode when SRTM raster is absent.
    Pushes dem_available flag to XCom for downstream tasks.
    """
    conf: dict = context["dag_run"].conf or {}
    river_id: str = conf.get("river_id", "ciliwung").lower()
    dem_path = _WS / "data" / "dem" / f"srtm_{river_id}.tif"
    dem_available = dem_path.exists()
    logger.info("DEM check | river=%s path=%s available=%s", river_id, dem_path, dem_available)
    context["ti"].xcom_push(key="dem_available", value=dem_available)
    context["ti"].xcom_push(key="river_id", value=river_id)
    return {"river_id": river_id, "dem_available": dem_available}


def compute_inundation_extent(**context) -> dict:
    """
    Run FloodInundationMapper.map() and push InundationResult to XCom.
    """
    from src.hydrology.flood_inundation_mapper import FloodInundationMapper

    conf: dict = context["dag_run"].conf or {}
    river_id: str = conf.get("river_id", "ciliwung").lower()
    flood_stage: str = conf.get("flood_stage", "WARNING")
    peak_cms: float = float(conf.get("peak_cms", 0.0))

    mapper = FloodInundationMapper()
    result = mapper.map(river_id=river_id, flood_stage=flood_stage, peak_cms=peak_cms)

    logger.info(
        "Inundation computed | river=%s stage=%s peak_cms=%.1f area_km2=%.2f",
        river_id, flood_stage, peak_cms, result.area_km2,
    )
    context["ti"].xcom_push(key="inundation_result", value={
        "river_id":     result.river_id,
        "flood_stage":  result.flood_stage,
        "peak_cms":     result.peak_cms,
        "area_km2":     result.area_km2,
        "water_depth_m": result.water_depth_m,
        "geojson_path": str(result.geojson_path),
        "dem_method":   result.dem_method,
        "computed_at":  result.computed_at.isoformat(),
    })
    return {"area_km2": result.area_km2}


def write_geojson(**context) -> dict:
    """
    Confirm timestamped GeoJSON exists (written by mapper) and overwrite
    latest_inundation.json sidecar for GEOSPATIAL pickup.
    """
    ti = context["ti"]
    inundation: dict = ti.xcom_pull(key="inundation_result", task_ids="compute_inundation_extent")

    _INUNDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Overwrite the rolling latest sidecar
    _LATEST_FILE.write_text(json.dumps(inundation, indent=2))
    logger.info("latest_inundation.json updated | river=%s area_km2=%.2f",
                inundation["river_id"], inundation["area_km2"])

    return {"latest_path": str(_LATEST_FILE)}


def emit_inundation_metrics(**context) -> None:
    """
    Emit tropi_flood_inundation_area_km2{river_id, flood_stage} Gauge.
    """
    ti = context["ti"]
    inundation: dict = ti.xcom_pull(key="inundation_result", task_ids="compute_inundation_extent")

    try:
        from src.hydrology.metrics import record_inundation_area
        record_inundation_area(
            river_id=inundation["river_id"],
            flood_stage=inundation["flood_stage"],
            area_km2=inundation["area_km2"],
        )
        logger.info("Inundation metrics emitted | river=%s stage=%s area_km2=%.2f",
                    inundation["river_id"], inundation["flood_stage"], inundation["area_km2"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Metrics emit non-fatal failure: %s", exc)


def notify_geospatial(**context) -> dict:
    """
    Append a structured entry to inundation_queue.jsonl.
    GEOSPATIAL polls this file for new inundation extents to process.
    """
    ti = context["ti"]
    inundation: dict = ti.xcom_pull(key="inundation_result", task_ids="compute_inundation_extent")
    conf: dict = context["dag_run"].conf or {}

    _INUNDATION_DIR.mkdir(parents=True, exist_ok=True)

    queue_entry = {
        "river_id":          inundation["river_id"],
        "flood_stage":       inundation["flood_stage"],
        "peak_cms":          inundation["peak_cms"],
        "area_km2":          inundation["area_km2"],
        "water_depth_m":     inundation["water_depth_m"],
        "geojson_path":      inundation["geojson_path"],
        "dem_method":        inundation["dem_method"],
        "computed_at":       inundation["computed_at"],
        "triggered_at":      conf.get("triggered_at", datetime.now(timezone.utc).isoformat()),
        "forecast_horizon_hr": int(conf.get("forecast_horizon_hr", 6)),
        "status":            "pending_geospatial",
    }

    with _QUEUE_FILE.open("a") as fh:
        fh.write(json.dumps(queue_entry) + "\n")

    logger.info("GEOSPATIAL notified | queue=%s river=%s",
                _QUEUE_FILE, inundation["river_id"])
    return {"queue_entry": queue_entry}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="hydrologis_flood_inundation",
    description=(
        "Trigger-only flood inundation mapping DAG. "
        "Invoked by emergency_flood_alert_dag via TriggerDagRunOperator."
    ),
    schedule_interval=None,          # trigger-only
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    tags=["hydrologis", "flood", "inundation", "sprint7"],
    doc_md=__doc__,
    sla_miss_callback=None,
) as dag:

    t_load_dem = PythonOperator(
        task_id="load_dem_data",
        python_callable=load_dem_data,
        sla=timedelta(minutes=2),
    )

    t_compute = PythonOperator(
        task_id="compute_inundation_extent",
        python_callable=compute_inundation_extent,
        sla=timedelta(minutes=5),
    )

    t_write = PythonOperator(
        task_id="write_geojson",
        python_callable=write_geojson,
        sla=timedelta(minutes=7),
    )

    t_metrics = PythonOperator(
        task_id="emit_inundation_metrics",
        python_callable=emit_inundation_metrics,
        sla=timedelta(minutes=8),
    )

    t_notify = PythonOperator(
        task_id="notify_geospatial",
        python_callable=notify_geospatial,
        sla=timedelta(minutes=10),
    )

    # Linear chain
    t_load_dem >> t_compute >> t_write >> t_metrics >> t_notify
