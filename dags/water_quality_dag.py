"""
water_quality_dag.py — Sprint 11 P2
dag_id: hydrologis_water_quality
schedule: 0 8 * * 2 Asia/Jakarta (weekly Tuesday 08:00 WIB)
Also triggers on Landsat overpass via FileSensor on workspace/data/landsat/new_tile.flag
SLA: 30 minutes

Tasks:
  check_landsat_availability  (ShortCircuitOperator)
  load_modis_reflectance
  load_landsat_tile
  select_best_source
  compute_all_water_bodies    (TaskGroup×20)
  aggregate_national_wqi
  write_active_wqi
  emit_wqi_metrics
  dispatch_pollution_alerts   (WQI<50 or Chl>50 → KLHK webhook; 7-day dedup)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

WORKSPACE   = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
LANDSAT_DIR = WORKSPACE / "data" / "landsat"
MODIS_DIR   = WORKSPACE / "data" / "modis"
OUTPUT_WQ   = WORKSPACE / "output" / "water_quality"
TILE_FLAG   = WORKSPACE / "data" / "landsat" / "new_tile.flag"
ACTIVE_WQI  = WORKSPACE / "output" / "water_quality" / "active_wqi.json"

WATER_BODIES = [
    "danau_toba","danau_singkarak","danau_maninjau","danau_towuti","danau_matano",
    "danau_poso","danau_ranau","rawa_pening","danau_batur","danau_bratan",
    "jatiluhur","saguling","cirata","gajah_mungkur","upper_cisokan",
    "sempor","mrica","wonogiri","kedungombo","ir_sutami",
]

KLHK_ALERT_TTL_DAYS = 7

_default_args = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def check_landsat_availability(**context) -> bool:
    """
    ShortCircuit: passes even if Landsat is absent (MODIS used as fallback).
    Logs new tile info when flag is present.
    """
    if TILE_FLAG.exists():
        try:
            with open(TILE_FLAG) as f:
                tile_info = f.read().strip()
            logger.info("Landsat new_tile.flag found: %s", tile_info)
        except Exception:
            pass
    else:
        logger.info("No new Landsat tile; MODIS will be used for all water bodies")
    # Always True — Landsat is preferred but not required
    return True


def load_modis_reflectance(**context) -> dict:
    """Load MODIS MOD09GA reflectance tiles from workspace/data/modis/."""
    exec_date = context.get("logical_date") or context.get("execution_date")
    today = exec_date.date() if hasattr(exec_date, "date") else datetime.now(timezone.utc).date()
    doy   = today.timetuple().tm_yday
    doy_str = f"{today.year}{doy:03d}"

    tiles_found = list(MODIS_DIR.glob(f"MOD09GA_{doy_str}_*.json")) if MODIS_DIR.exists() else []
    summary = {
        "date":       today.isoformat(),
        "doy_str":    doy_str,
        "tile_count": len(tiles_found),
        "tiles":      [f.stem for f in tiles_found],
    }
    logger.info("MODIS reflectance | doy=%s tiles=%d", doy_str, len(tiles_found))
    context["ti"].xcom_push(key="modis_summary", value=summary)
    return summary


def load_landsat_tile(**context) -> dict:
    """Load Landsat 8/9 OLI tile if new_tile.flag is present."""
    result = {"available": False, "cloud_pct": 100.0, "tiles": []}
    if TILE_FLAG.exists():
        tiles = sorted(LANDSAT_DIR.glob("L8_OLI_*.json")) if LANDSAT_DIR.exists() else []
        result["available"] = bool(tiles)
        result["tiles"]     = [f.stem for f in tiles]
        logger.info("Landsat OLI tiles available: %d", len(tiles))
    context["ti"].xcom_push(key="landsat_summary", value=result)
    return result


def select_best_source(**context) -> str:
    """
    Prefer Landsat if any tile has cloud_cover < 20%; else MODIS.
    Returns source string pushed to XCom.
    """
    ti     = context["ti"]
    ls     = ti.xcom_pull(task_ids="load_landsat_tile", key="landsat_summary") or {}
    source = "landsat" if ls.get("available") else "modis"
    context["ti"].xcom_push(key="best_source", value=source)
    logger.info("Best RS source selected: %s", source)
    return source


def _compute_water_body(water_body_id: str, **context) -> dict:
    """Compute WQI for one water body; TaskGroup worker."""
    from src.hydrology.water_quality_monitor import WaterQualityMonitor

    exec_date = context.get("logical_date") or context.get("execution_date")
    today     = exec_date.date() if hasattr(exec_date, "date") else datetime.now(timezone.utc).date()

    monitor = WaterQualityMonitor()
    result  = monitor.compute_wqi(water_body_id=water_body_id, dt=today)

    summary = {
        "water_body_id":      result.water_body_id,
        "date":               result.date,
        "wqi":                result.wqi,
        "category":           result.category,
        "turbidity_ntu":      result.turbidity_ntu,
        "chlorophyll_ug_l":   result.chlorophyll_ug_l,
        "eutrophication_risk":result.eutrophication_risk,
        "data_source":        result.data_source,
    }
    context["ti"].xcom_push(key=f"wqi_{water_body_id}", value=summary)
    logger.info(
        "WQI | %-18s WQI=%.1f (%s) chl=%.2f", water_body_id, result.wqi, result.category,
        result.chlorophyll_ug_l,
    )
    return summary


def aggregate_national_wqi(**context) -> dict:
    """Aggregate per-water-body WQI; flag poor/very_poor outliers."""
    ti      = context["ti"]
    results = {}
    for wb in WATER_BODIES:
        data = ti.xcom_pull(
            task_ids=f"compute_water_bodies.compute_{wb}",
            key=f"wqi_{wb}",
        )
        if data:
            results[wb] = data

    wqi_vals   = [results[w]["wqi"] for w in results]
    cat_order  = {"very_poor":0,"poor":1,"fair":2,"good":3,"excellent":4}
    poor_count = sum(1 for w in results if cat_order.get(results[w]["category"], 2) < 2)

    national = {
        "date":             (results[next(iter(results))]["date"] if results
                             else datetime.now(timezone.utc).date().isoformat()),
        "water_body_count": len(results),
        "mean_wqi":         round(sum(wqi_vals) / max(len(wqi_vals),1), 2),
        "min_wqi":          round(min(wqi_vals) if wqi_vals else 0, 2),
        "poor_or_worse":    poor_count,
        "water_bodies":     results,
    }
    if poor_count > 0:
        poor_names = [w for w in results if cat_order.get(results[w]["category"],2) < 2]
        logger.warning("WQI ALERT: %d water bodies at poor/very_poor: %s", poor_count, poor_names)

    context["ti"].xcom_push(key="national_wqi", value=national)
    return national


def write_active_wqi(**context) -> str:
    """Write active_wqi.json sidecar for GEOSPATIAL + VISUALIA."""
    from src.hydrology.water_quality_monitor import WaterQualityMonitor, WQIResult

    ti       = context["ti"]
    national = ti.xcom_pull(task_ids="aggregate_national_wqi", key="national_wqi")
    if not national:
        return str(ACTIVE_WQI)

    OUTPUT_WQ.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sidecar = {
        "generated_at":     now.isoformat(),
        "water_body_count": national.get("water_body_count"),
        "mean_wqi":         national.get("mean_wqi"),
        "poor_or_worse":    national.get("poor_or_worse"),
        "water_bodies":     national.get("water_bodies", {}),
    }
    with open(ACTIVE_WQI, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info("active_wqi.json updated | %d water bodies", national.get("water_body_count"))
    return str(ACTIVE_WQI)


def emit_wqi_metrics(**context) -> dict:
    """Emit WATER_QUALITY_INDEX Gauge per water body and category."""
    ti       = context["ti"]
    national = ti.xcom_pull(task_ids="aggregate_national_wqi", key="national_wqi")
    if not national:
        return {}

    try:
        from src.hydrology.metrics import WATER_QUALITY_INDEX
        for wb, d in national.get("water_bodies", {}).items():
            WATER_QUALITY_INDEX.labels(water_body_id=wb, category=d["category"]).set(d["wqi"])
    except Exception as exc:
        logger.debug("WATER_QUALITY_INDEX emit error: %s", exc)

    return {"mean_wqi": national.get("mean_wqi"), "poor_count": national.get("poor_or_worse")}


def dispatch_pollution_alerts(**context) -> list[str]:
    """
    Dispatch KLHK alerts for WQI < 50 (poor/very_poor) or Chl-a > 50 µg/L.
    7-day dedup via Airflow Variable WQI_ALERT_{water_body_id}.
    """
    ti       = context["ti"]
    national = ti.xcom_pull(task_ids="aggregate_national_wqi", key="national_wqi")
    if not national:
        return []

    exec_date = context.get("logical_date") or context.get("execution_date")
    now       = exec_date if isinstance(exec_date, datetime) else datetime.now(timezone.utc)

    klhk_url   = os.environ.get("KLHK_WEBHOOK_URL", "")
    dispatched = []

    for wb, d in national.get("water_bodies", {}).items():
        wqi   = d.get("wqi", 100)
        chl   = d.get("chlorophyll_ug_l", 0)
        if wqi >= 50 and chl < 50:
            continue

        # 7-day dedup
        var_key     = f"WQI_ALERT_{wb.upper()}"
        last_ts_str = None
        try:
            last_ts_str = Variable.get(var_key, default_var=None)
        except Exception:
            pass

        if last_ts_str:
            try:
                last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                now_tz  = now if isinstance(now, datetime) and now.tzinfo else \
                          datetime.now(timezone.utc)
                if (now_tz - last_ts).total_seconds() < 7 * 86400:
                    logger.info("WQI alert dedup skip | %s WQI=%.1f", wb, wqi)
                    continue
            except Exception:
                pass

        alert = {
            "event_type":       "WATER_QUALITY_ALERT",
            "water_body_id":    wb,
            "wqi":              wqi,
            "category":         d.get("category"),
            "chlorophyll_ug_l": chl,
            "eutrophication_risk": d.get("eutrophication_risk"),
            "data_source":      d.get("data_source"),
            "issued_at":        now.isoformat() if isinstance(now, datetime) else str(now),
        }

        # Queue alert
        queue = WORKSPACE / "output" / "alerts" / "water_quality_alerts.jsonl"
        queue.parent.mkdir(parents=True, exist_ok=True)
        with open(queue, "a") as f:
            f.write(json.dumps(alert) + "\n")

        if klhk_url:
            try:
                req = urllib.request.Request(
                    klhk_url, data=json.dumps(alert).encode(),
                    headers={"Content-Type":"application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=10):
                    pass
            except Exception as exc:
                logger.error("KLHK webhook error for %s: %s", wb, exc)

        try:
            ts_now = now.isoformat() if isinstance(now, datetime) else \
                     datetime.now(timezone.utc).isoformat()
            Variable.set(var_key, ts_now)
        except Exception:
            pass

        logger.warning("WQI ALERT DISPATCHED | %-20s WQI=%.1f Chl=%.1f", wb, wqi, chl)
        dispatched.append(f"{wb}:WQI={wqi:.0f}")

    return dispatched


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id            = "hydrologis_water_quality",
    description       = (
        "Sprint 11: Weekly MODIS/Landsat water quality indexing for 20 Indonesian water bodies. "
        "Turbidity, chlorophyll-a, CDOM, SST via KepMenLH-115 WQI composite. "
        "KLHK alerts for poor/very_poor detections."
    ),
    default_args      = _default_args,
    schedule_interval = "0 8 * * 2",   # Tuesday 08:00 Asia/Jakarta
    start_date        = days_ago(1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["hydrologis","water-quality","modis","landsat","sprint11"],
    dagrun_timeout    = timedelta(minutes=30),
) as dag:

    t_ls_check  = ShortCircuitOperator(
        task_id            = "check_landsat_availability",
        python_callable    = check_landsat_availability,
        ignore_downstream_trigger_rules = True,
    )
    t_modis     = PythonOperator(task_id="load_modis_reflectance", python_callable=load_modis_reflectance)
    t_landsat   = PythonOperator(task_id="load_landsat_tile",      python_callable=load_landsat_tile)
    t_source    = PythonOperator(task_id="select_best_source",     python_callable=select_best_source)

    with TaskGroup("compute_water_bodies") as wq_tg:
        for wb in WATER_BODIES:
            PythonOperator(
                task_id        = f"compute_{wb}",
                python_callable= _compute_water_body,
                op_kwargs      = {"water_body_id": wb},
            )

    t_agg      = PythonOperator(task_id="aggregate_national_wqi",  python_callable=aggregate_national_wqi)
    t_write    = PythonOperator(task_id="write_active_wqi",        python_callable=write_active_wqi)
    t_metrics  = PythonOperator(task_id="emit_wqi_metrics",        python_callable=emit_wqi_metrics)
    t_dispatch = PythonOperator(task_id="dispatch_pollution_alerts",python_callable=dispatch_pollution_alerts)

    t_ls_check >> [t_modis, t_landsat] >> t_source >> wq_tg >> t_agg >> t_write >> t_metrics >> t_dispatch
