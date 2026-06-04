"""
Airflow DAG — QPE Output Validation & Ingest (30-minute)
HYDROLOGIS Sprint 5 | Deliverable 2

dag_id:   hydrologis_qpe_ingest
schedule: */30 * * * * (Asia/Jakarta) — aligned to flood_early_warning_dag
SLA:      10 minutes

Reads QPE output from workspace/output/qpe/ (latest file by mtime).
Validates:
  1. File age < 35 min (SLA guard)
  2. Non-null precip grid
  3. Coverage ≥ 80% of domain

On validation failure: skip downstream, increment
  tropi_qpe_validation_failures_total{reason}

On success: write workspace/data/qpe_latest_validated.json
  Fields: precip_mm_6h, sm_m3m3, timestamp_utc
  Consumed by: flood_early_warning_dag → fetch_qpe task

Prometheus:
  tropi_qpe_last_validated_timestamp_seconds   Gauge
  tropi_qpe_validation_failures_total{reason}  Counter
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("WORKSPACE_ROOT", "workspace")
QPE_DIR            = os.path.join(WORKSPACE, "output", "qpe")
QPE_VALIDATED_PATH = os.path.join(WORKSPACE, "data", "qpe_latest_validated.json")

QPE_MAX_AGE_MINUTES  = 35     # SLA guard: file must be < 35 min old
QPE_MIN_COVERAGE_PCT = 80.0   # Domain coverage threshold

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
# Task 1: find_latest_qpe
# ---------------------------------------------------------------------------

def find_latest_qpe(**context) -> dict:
    """
    Locate the most-recently modified QPE JSON output in workspace/output/qpe/.
    Returns file metadata; raises if directory empty.
    """
    import glob

    os.makedirs(QPE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(QPE_DIR, "*.json")), key=os.path.getmtime, reverse=True)

    if not files:
        logger.warning("No QPE output files found in %s", QPE_DIR)
        return {"found": False, "path": None, "mtime": None}

    latest = files[0]
    mtime  = os.path.getmtime(latest)
    age_s  = time.time() - mtime

    logger.info(
        "Latest QPE file: %s | age=%.0fs (%.1f min)",
        latest, age_s, age_s / 60,
    )
    return {
        "found": True,
        "path":  latest,
        "mtime": mtime,
        "age_s": age_s,
    }


# ---------------------------------------------------------------------------
# Task 2: validate_qpe
# ---------------------------------------------------------------------------

def validate_qpe(**context) -> dict:
    """
    Validate QPE file against 3 checks:
      1. age < 35 min
      2. precip grid not null
      3. coverage >= 80%
    On failure: increment tropi_qpe_validation_failures_total{reason}
    """
    from src.hydrology.metrics import QPE_VALIDATED_TS, QPE_VALIDATION_FAILURES, push_metrics

    ti       = context["ti"]
    file_meta = ti.xcom_pull(task_ids="find_latest_qpe")

    if not file_meta.get("found"):
        QPE_VALIDATION_FAILURES.labels(reason="no_file").inc()
        push_metrics()
        return {"valid": False, "reason": "no_file"}

    path  = file_meta["path"]
    age_s = file_meta["age_s"]

    # Check 1: Age
    if age_s > QPE_MAX_AGE_MINUTES * 60:
        age_min = age_s / 60
        logger.warning(
            "QPE file too old | age=%.1f min (limit=%d min) path=%s",
            age_min, QPE_MAX_AGE_MINUTES, path,
        )
        QPE_VALIDATION_FAILURES.labels(reason="file_too_old").inc()
        push_metrics()
        return {"valid": False, "reason": "file_too_old", "age_min": age_min}

    # Load file
    try:
        with open(path) as fh:
            qpe_data = json.load(fh)
    except Exception as exc:
        logger.error("QPE file parse error: %s", exc)
        QPE_VALIDATION_FAILURES.labels(reason="parse_error").inc()
        push_metrics()
        return {"valid": False, "reason": "parse_error", "error": str(exc)}

    # Check 2: Non-null precip grid
    precip = qpe_data.get("precip_mm_6h") or qpe_data.get("mean_precip_mm")
    if precip is None:
        logger.warning("QPE file has null precip field | path=%s", path)
        QPE_VALIDATION_FAILURES.labels(reason="null_precip").inc()
        push_metrics()
        return {"valid": False, "reason": "null_precip"}

    # Check 3: Domain coverage >= 80%
    coverage_pct = float(qpe_data.get("coverage_pct", 100.0))
    if coverage_pct < QPE_MIN_COVERAGE_PCT:
        logger.warning(
            "QPE domain coverage insufficient | coverage=%.1f%% (min=%.0f%%)",
            coverage_pct, QPE_MIN_COVERAGE_PCT,
        )
        QPE_VALIDATION_FAILURES.labels(reason="insufficient_coverage").inc()
        push_metrics()
        return {"valid": False, "reason": "insufficient_coverage", "coverage_pct": coverage_pct}

    # All checks passed
    QPE_VALIDATED_TS.set(time.time())
    logger.info(
        "QPE validation passed | precip=%.1f mm/6h coverage=%.1f%% age=%.1f min",
        precip, coverage_pct, age_s / 60,
    )
    push_metrics()

    return {
        "valid":         True,
        "path":          path,
        "precip_mm_6h":  float(precip),
        "sm_m3m3":       float(qpe_data.get("sm_m3m3", 0.0)),
        "coverage_pct":  coverage_pct,
        "timestamp_utc": qpe_data.get(
            "timestamp_utc",
            datetime.fromtimestamp(file_meta["mtime"], tz=timezone.utc).isoformat(),
        ),
    }


# ---------------------------------------------------------------------------
# Task 3: write_validated_output  (skipped when validation fails)
# ---------------------------------------------------------------------------

def write_validated_output(**context) -> Optional[str]:
    """
    Write workspace/data/qpe_latest_validated.json when validation passed.
    If validation failed, log and return None (downstream tasks use old file).
    """
    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="validate_qpe")

    if not result.get("valid"):
        logger.info(
            "QPE validation failed (%s) — skipping write, flood_early_warning_dag will use previous validated file",
            result.get("reason"),
        )
        return None

    payload = {
        "precip_mm_6h":  result["precip_mm_6h"],
        "sm_m3m3":       result.get("sm_m3m3", 0.0),
        "coverage_pct":  result.get("coverage_pct", 100.0),
        "timestamp_utc": result["timestamp_utc"],
        "validated_at":  datetime.now(timezone.utc).isoformat(),
        "source_path":   result["path"],
    }

    os.makedirs(os.path.dirname(QPE_VALIDATED_PATH), exist_ok=True)
    with open(QPE_VALIDATED_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        "QPE validated output written | precip=%.1f mm/6h path=%s",
        payload["precip_mm_6h"], QPE_VALIDATED_PATH,
    )
    return QPE_VALIDATED_PATH


# ---------------------------------------------------------------------------
# Task 4: emit_qpe_metrics
# ---------------------------------------------------------------------------

def emit_qpe_metrics(**context) -> None:
    """Confirm metric push complete (push_metrics already called in validate_qpe)."""
    from src.hydrology.metrics import push_metrics

    ti     = context["ti"]
    result = ti.xcom_pull(task_ids="validate_qpe")
    push_metrics()
    logger.info(
        "QPE metrics emitted | valid=%s reason=%s",
        result.get("valid"), result.get("reason", "ok"),
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="hydrologis_qpe_ingest",
    description="30-minute QPE output validation and ingest — feeds flood_early_warning_30min fetch_qpe task",
    schedule="*/30 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["hydrologis", "qpe", "validation", "realtime"],
    doc_md="""
## HYDROLOGIS — QPE Output Validation & Ingest (30-minute)

**Schedule:** Every 30 minutes (Asia/Jakarta) — aligned to `flood_early_warning_30min`  
**SLA:** 10 minutes  
**Input:** `workspace/output/qpe/` (latest JSON by mtime, from QPEFusionPipeline)  
**Output:** `workspace/data/qpe_latest_validated.json` (consumed by `flood_early_warning_30min`)

**Validation checks:**
1. File age < 35 min (SLA guard)
2. `precip_mm_6h` / `mean_precip_mm` not null
3. Coverage ≥ 80% of domain

**On failure:** skip write + increment `tropi_qpe_validation_failures_total{reason}`  
**Prometheus:** `tropi_qpe_last_validated_timestamp_seconds`
    """,
    dagrun_timeout=timedelta(minutes=10),
) as dag:

    t_find = PythonOperator(
        task_id="find_latest_qpe",
        python_callable=find_latest_qpe,
    )

    t_validate = PythonOperator(
        task_id="validate_qpe",
        python_callable=validate_qpe,
    )

    t_write = PythonOperator(
        task_id="write_validated_output",
        python_callable=write_validated_output,
    )

    t_metrics = PythonOperator(
        task_id="emit_qpe_metrics",
        python_callable=emit_qpe_metrics,
    )

    t_find >> t_validate >> t_write >> t_metrics
