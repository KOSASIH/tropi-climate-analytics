"""
qpe_pipeline_dag.py — Sprint 8 J2
dag_id: hydrologis_qpe_pipeline
schedule: */30 * * * * Asia/Jakarta (every 30 minutes)
SLA: 8 minutes from schedule time

Tasks:
  fetch_gpm_data      — download/stub IMERG_{valid_time}.nc
  fetch_bmkg_gauges   — download/stub gauges_{date}.csv
  run_qpe_merge       — QPEPipeline.run(valid_time) → XCom QPEResult
  write_output        — confirm latest_qpe.json
  emit_qpe_metrics    — QPE_UPDATE_LATENCY + max_precip Prometheus
  notify_downstream   — append to qpe_queue.jsonl + set QPE_EXTREME_EVENT if ≥50mm
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

WORKSPACE  = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
GPM_DIR    = WORKSPACE / "data" / "gpm"
BMKG_DIR   = WORKSPACE / "data" / "bmkg"
OUTPUT_DIR = WORKSPACE / "output" / "qpe"
SIDECAR    = WORKSPACE / "output" / "qpe" / "latest_qpe.json"
QUEUE      = WORKSPACE / "output" / "qpe" / "qpe_queue.jsonl"

_EXTREME_THRESHOLD_MM = 50.0

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

def fetch_gpm_data(**context) -> dict:
    """
    Download or stub GPM IMERG half-hourly file for valid_time.
    In production: call NASA GES DISC API or S3 bucket.
    Stub: creates synthetic placeholder metadata file.
    """
    exec_date = context.get("logical_date") or context.get("execution_date")
    if exec_date is None:
        exec_date = datetime.now(timezone.utc)
    valid_time = exec_date if hasattr(exec_date, "strftime") else \
                 datetime.fromisoformat(str(exec_date))

    ts_str   = valid_time.strftime("%Y%m%d_%H%M")
    nc_path  = GPM_DIR / f"IMERG_{ts_str}.nc"
    GPM_DIR.mkdir(parents=True, exist_ok=True)

    if not nc_path.exists():
        # Stub: write JSON metadata placeholder (real impl would fetch .nc)
        stub = {
            "source":     "stub",
            "valid_time": valid_time.isoformat(),
            "note":       "Real IMERG NetCDF not available; QPEPipeline will use synthetic field",
        }
        with open(nc_path.with_suffix(".json"), "w") as f:
            json.dump(stub, f)
        logger.info("GPM stub metadata written | valid_time=%s", ts_str)
        gpm_available = False
    else:
        gpm_available = True
        logger.info("GPM IMERG file found | %s", nc_path)

    context["ti"].xcom_push(key="valid_time_iso", value=valid_time.isoformat())
    context["ti"].xcom_push(key="gpm_available",  value=gpm_available)
    return {"valid_time": valid_time.isoformat(), "gpm_available": gpm_available}


def fetch_bmkg_gauges(**context) -> dict:
    """
    Download or stub BMKG gauge CSV for today's date.
    In production: fetch from BMKG API endpoint.
    Stub: creates synthetic gauge CSV for testing.
    """
    ti         = context["ti"]
    vt_iso     = ti.xcom_pull(task_ids="fetch_gpm_data", key="valid_time_iso")
    valid_time = datetime.fromisoformat(vt_iso) if vt_iso else datetime.now(timezone.utc)
    date_str   = valid_time.strftime("%Y%m%d")

    BMKG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = BMKG_DIR / f"gauges_{date_str}.csv"

    if not csv_path.exists():
        # Stub: synthetic gauge network (10 representative stations)
        import random
        rng      = random.Random(int(valid_time.timestamp()))
        stations = [
            ("BMK001",  -6.2,  106.8),  # Bogor (Ciliwung)
            ("BMK002",  -6.4,  106.9),  # Depok
            ("BMK003",  -6.9,  107.6),  # Bandung
            ("BMK004",  -7.2,  112.7),  # Malang (Brantas)
            ("BMK005",  -7.6,  112.2),  # Kediri
            ("BMK006",  -7.5,  111.5),  # Nganjuk
            ("BMK007",  -7.0,  110.4),  # Semarang (Solo)
            ("BMK008",  -7.5,  110.8),  # Purwodadi
            ("BMK009",  -7.6,  111.0),  # Blora
            ("BMK010",  -7.2,  111.9),  # Ngawi
        ]
        rows = ["station_id,lat,lon,precip_mm_30min"]
        for sid, lat, lon in stations:
            precip = round(max(rng.gauss(3.0, 4.0), 0.0), 2)
            rows.append(f"{sid},{lat},{lon},{precip}")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")
        logger.info("BMKG gauge stub written | %d stations date=%s", len(stations), date_str)

    return {"gauge_path": str(csv_path), "date": date_str}


def run_qpe_merge(**context) -> dict:
    """Run QPEPipeline.run() and push QPEResult to XCom."""
    from src.hydrology.qpe_pipeline import QPEPipeline

    ti     = context["ti"]
    vt_iso = ti.xcom_pull(task_ids="fetch_gpm_data", key="valid_time_iso")
    if vt_iso:
        valid_time = datetime.fromisoformat(vt_iso)
    else:
        exec_date  = context.get("logical_date") or context.get("execution_date")
        valid_time = exec_date if isinstance(exec_date, datetime) else datetime.now(timezone.utc)

    pipeline = QPEPipeline()
    result   = pipeline.run(valid_time=valid_time)

    summary = {
        "valid_time":        result.valid_time,
        "num_gauges_merged": result.num_gauges_merged,
        "max_precip_mm":     result.max_precip_mm,
        "mean_precip_mm":    result.mean_precip_mm,
        "geojson_path":      result.geojson_path,
        "sidecar_path":      result.sidecar_path,
        "method":            result.method,
        "gpm_latency_s":     result.gpm_latency_s,
        "merge_latency_s":   result.merge_latency_s,
        "warnings":          result.warnings,
    }

    ti.xcom_push(key="qpe_result", value=summary)
    logger.info(
        "QPE merge complete | valid=%s method=%s max=%.1f mm gauges=%d",
        result.valid_time, result.method, result.max_precip_mm, result.num_gauges_merged,
    )
    return summary


def write_output(**context) -> str:
    """Confirm latest_qpe.json sidecar is written and log its contents."""
    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="run_qpe_merge", key="qpe_result")

    if not result:
        logger.warning("No QPE result in XCom; skipping output confirmation")
        return "no_result"

    sidecar = result.get("sidecar_path", str(SIDECAR))
    if Path(sidecar).exists():
        logger.info("QPE sidecar confirmed | path=%s max=%.1f mm",
                    sidecar, result.get("max_precip_mm", 0.0))
        return sidecar
    else:
        logger.error("QPE sidecar NOT found at %s", sidecar)
        return "missing"


def emit_qpe_metrics(**context) -> dict:
    """Log QPE summary and emit Prometheus QPE_UPDATE_LATENCY."""
    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="run_qpe_merge", key="qpe_result")

    if not result:
        return {}

    try:
        from src.hydrology.metrics import QPE_UPDATE_LATENCY
        # Total pipeline latency
        total_s = result.get("gpm_latency_s", 0.0) + result.get("merge_latency_s", 0.0)
        QPE_UPDATE_LATENCY.labels(source="merged").observe(total_s)
    except Exception as exc:
        logger.debug("QPE_UPDATE_LATENCY emit error: %s", exc)

    logger.info(
        "QPE metrics | valid=%s max=%.1f mean=%.2f mm method=%s",
        result.get("valid_time"),
        result.get("max_precip_mm", 0.0),
        result.get("mean_precip_mm", 0.0),
        result.get("method"),
    )
    return {
        "max_precip_mm":   result.get("max_precip_mm"),
        "mean_precip_mm":  result.get("mean_precip_mm"),
        "method":          result.get("method"),
    }


def notify_downstream(**context) -> str:
    """
    Append QPE event to qpe_queue.jsonl (GEOSPATIAL pickup).
    If max_precip ≥ 50mm/30min, set Airflow Variable QPE_EXTREME_EVENT=true.
    """
    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="run_qpe_merge", key="qpe_result")

    if not result:
        logger.warning("No QPE result; skipping downstream notification")
        return "skipped"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "event_ts":          datetime.now(timezone.utc).isoformat(),
        "valid_time":        result.get("valid_time"),
        "max_precip_mm":     result.get("max_precip_mm"),
        "mean_precip_mm":    result.get("mean_precip_mm"),
        "num_gauges_merged": result.get("num_gauges_merged"),
        "method":            result.get("method"),
        "geojson_path":      result.get("geojson_path"),
        "sidecar_path":      result.get("sidecar_path"),
    }

    with open(QUEUE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Extreme event flag
    max_p = float(result.get("max_precip_mm", 0.0))
    if max_p >= _EXTREME_THRESHOLD_MM:
        try:
            Variable.set("QPE_EXTREME_EVENT", "true")
            logger.warning(
                "EXTREME QPE EVENT: max=%.1f mm ≥ %.0f mm threshold — "
                "QPE_EXTREME_EVENT=true set",
                max_p, _EXTREME_THRESHOLD_MM,
            )
        except Exception as exc:
            logger.error("Failed to set QPE_EXTREME_EVENT Variable: %s", exc)
    else:
        # Reset flag if below threshold
        try:
            Variable.set("QPE_EXTREME_EVENT", "false")
        except Exception:
            pass

    logger.info(
        "GEOSPATIAL notified | queue=%s valid=%s max=%.1f mm extreme=%s",
        QUEUE,
        result.get("valid_time"),
        max_p,
        max_p >= _EXTREME_THRESHOLD_MM,
    )
    return str(QUEUE)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id            = "hydrologis_qpe_pipeline",
    description       = (
        "Sprint 8: GPM IMERG + BMKG gauge OI merge → 4km/30-min QPE rasters. "
        "Feeds GEOSPATIAL and emergency_flood_alert_dag."
    ),
    default_args      = _default_args,
    schedule_interval = "*/30 * * * *",
    start_date        = days_ago(1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["hydrologis", "qpe", "precipitation", "sprint8"],
    dagrun_timeout    = timedelta(minutes=8),
) as dag:

    t_gpm     = PythonOperator(task_id="fetch_gpm_data",    python_callable=fetch_gpm_data)
    t_bmkg    = PythonOperator(task_id="fetch_bmkg_gauges", python_callable=fetch_bmkg_gauges)
    t_merge   = PythonOperator(task_id="run_qpe_merge",     python_callable=run_qpe_merge)
    t_write   = PythonOperator(task_id="write_output",      python_callable=write_output)
    t_metrics = PythonOperator(task_id="emit_qpe_metrics",  python_callable=emit_qpe_metrics)
    t_notify  = PythonOperator(task_id="notify_downstream", python_callable=notify_downstream)

    [t_gpm, t_bmkg] >> t_merge >> t_write >> t_metrics >> t_notify
