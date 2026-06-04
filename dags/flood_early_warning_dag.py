"""
flood_early_warning_dag.py — Sprint 9 K2
dag_id: hydrologis_flood_early_warning

Schedule: */30 * * * * Asia/Jakarta  (every 30 min — matches QPE cadence)
SLA: 5 minutes

Integrates QPE latest_qpe.json + streamflow latest_forecast.json for 6 rivers,
evaluates BNPB/BPBD 4-level flood warnings, writes active_warnings.json sidecar,
emits FLOOD_ALERT_DISPATCH Prometheus counter, dispatches BPBD webhook alerts
for ORANGE/RED with 30-minute dedup window.

Rivers: ciliwung, brantas, solo, citarum, musi, bengawan_solo

Tasks:
    load_qpe_latest           → ShortCircuit if QPE stale > 45 min
    load_streamflow_latest    → ShortCircuit if forecast stale > 75 min
    evaluate_all_rivers       (TaskGroup, 6 parallel)
      ├─ evaluate_ciliwung
      ├─ evaluate_brantas
      ├─ evaluate_solo
      ├─ evaluate_citarum
      ├─ evaluate_musi
      └─ evaluate_bengawan_solo
    write_active_warnings     → active_warnings.json + per-river JSONs
    emit_warning_metrics      → FLOOD_ALERT_DISPATCH Counter
    dispatch_alerts           → BPBD webhook for ORANGE/RED (30-min dedup)

Cross-agent handoff:
    GEOSPATIAL: workspace/output/early_warning/active_warnings.json
    VISUALIA:   workspace/output/early_warning/active_warnings.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "hydrologis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
}

_WS            = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_QPE_PATH      = _WS / "output" / "qpe"   / "latest_qpe.json"
_SF_PATH       = _WS / "output" / "streamflow" / "latest_forecast.json"
_WARN_DIR      = _WS / "output" / "early_warning"
_QPE_STALE_S   = 45 * 60   # 45 minutes
_SF_STALE_S    = 75 * 60   # 75 minutes
_RIVERS        = ["ciliwung", "brantas", "solo", "citarum", "musi", "bengawan_solo"]
_ALERT_LEVELS  = {"ORANGE", "RED"}
_DEDUP_TTL_S   = 30 * 60   # 30-minute alert dedup window


# ── staleness guards ──────────────────────────────────────────────────────────

def _check_file_freshness(path: Path, max_age_s: int) -> bool:
    if not path.exists():
        logger.warning("File missing — ShortCircuit: %s", path)
        return False
    age = (datetime.now(tz=timezone.utc).timestamp() -
           path.stat().st_mtime)
    if age > max_age_s:
        logger.warning("File stale (%.0fs > %ds) — ShortCircuit: %s", age, max_age_s, path)
        return False
    return True


def load_qpe_latest(**context) -> bool:
    ok = _check_file_freshness(_QPE_PATH, _QPE_STALE_S)
    if ok:
        data = json.loads(_QPE_PATH.read_text())
        context["ti"].xcom_push(key="qpe_data", value=data)
    return ok


def load_streamflow_latest(**context) -> bool:
    ok = _check_file_freshness(_SF_PATH, _SF_STALE_S)
    if ok:
        data = json.loads(_SF_PATH.read_text())
        context["ti"].xcom_push(key="streamflow_data", value=data)
    return ok


# ── per-river evaluation factory ─────────────────────────────────────────────

def _make_evaluate_callable(river_id: str):
    def evaluate_river(**context):
        from src.hydrology.flood_early_warning import FloodEarlyWarningSystem
        ti = context["ti"]
        qpe  = ti.xcom_pull(key="qpe_data",         task_ids="load_qpe_latest")
        sf   = ti.xcom_pull(key="streamflow_data",   task_ids="load_streamflow_latest")
        ews  = FloodEarlyWarningSystem()
        result = ews.evaluate(
            river_id=river_id,
            issue_time=datetime.now(tz=timezone.utc),
            qpe_data=qpe,
            streamflow_data=sf,
        )
        payload = {
            "river_id":            result.river_id,
            "issue_time":          result.issue_time.isoformat(),
            "level":               result.level,
            "forecast_cms_6hr":    result.forecast_cms_6hr,
            "forecast_cms_12hr":   result.forecast_cms_12hr,
            "forecast_cms_24hr":   result.forecast_cms_24hr,
            "flood_threshold_cms": result.flood_threshold_cms,
            "confidence_pct":      result.confidence_pct,
            "expected_peak_time":  result.expected_peak_time.isoformat()
                                   if result.expected_peak_time else None,
            "alerted":             result.alerted,
        }
        ti.xcom_push(key=f"warning_{river_id}", value=payload)
        logger.info("Flood eval | %s level=%s %.0f/%.0f cms",
                    river_id, result.level,
                    result.forecast_cms_6hr, result.flood_threshold_cms)
        return payload
    evaluate_river.__name__ = f"evaluate_{river_id}"
    return evaluate_river


# ── downstream tasks ──────────────────────────────────────────────────────────

def write_active_warnings(**context) -> dict:
    ti = context["ti"]
    warnings = []
    for river_id in _RIVERS:
        w = ti.xcom_pull(
            key=f"warning_{river_id}",
            task_ids=f"evaluate_all_rivers.evaluate_{river_id}",
        )
        if w:
            warnings.append(w)

    _WARN_DIR.mkdir(parents=True, exist_ok=True)

    # Per-river files
    for w in warnings:
        ts = datetime.fromisoformat(w["issue_time"]).strftime("%Y%m%d_%H%M")
        path = _WARN_DIR / f"warning_{w['river_id']}_{ts}.json"
        path.write_text(json.dumps(w, indent=2))

    # Rolling sidecar
    active = [w for w in warnings if w["level"] in ("YELLOW", "ORANGE", "RED")]
    sidecar = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "active_warnings": active,
        "total_rivers_monitored": len(_RIVERS),
        "rivers_at_risk": len(active),
        "highest_level": max((w["level"] for w in active),
                              key=lambda l: ("GREEN","YELLOW","ORANGE","RED").index(l))
                          if active else "GREEN",
    }
    (_WARN_DIR / "active_warnings.json").write_text(json.dumps(sidecar, indent=2))
    logger.info("active_warnings.json written | rivers_at_risk=%d", len(active))
    ti.xcom_push(key="warnings", value=warnings)
    return {"rivers_at_risk": len(active)}


def emit_warning_metrics(**context) -> None:
    ti = context["ti"]
    warnings = ti.xcom_pull(key="warnings", task_ids="write_active_warnings") or []
    try:
        from src.hydrology.metrics import FLOOD_ALERT_DISPATCH
        for w in warnings:
            if w["level"] != "GREEN":
                FLOOD_ALERT_DISPATCH.labels(
                    river_id=w["river_id"], level=w["level"]
                ).inc()
        logger.info("FLOOD_ALERT_DISPATCH metrics emitted for %d rivers", len(warnings))
    except Exception as exc:
        logger.warning("Metrics emit non-fatal: %s", exc)


def dispatch_alerts(**context) -> dict:
    """
    Push BPBD webhook alerts for ORANGE/RED rivers.
    Dedup: Airflow Variable FLOOD_ALERT_DISPATCHED_{river_id} stores last dispatch
    timestamp; skip re-dispatch within _DEDUP_TTL_S (30 min).
    """
    ti = context["ti"]
    warnings = ti.xcom_pull(key="warnings", task_ids="write_active_warnings") or []
    alert_rivers = [w for w in warnings if w["level"] in _ALERT_LEVELS]
    dispatched = []

    for w in alert_rivers:
        var_key  = f"FLOOD_ALERT_DISPATCHED_{w['river_id'].upper()}"
        last_str = Variable.get(var_key, default_var=None)
        now_ts   = datetime.now(tz=timezone.utc).timestamp()

        if last_str and (now_ts - float(last_str)) < _DEDUP_TTL_S:
            logger.info("Dedup suppressed | %s last=%.0fs ago",
                        w["river_id"], now_ts - float(last_str))
            continue

        try:
            from src.hydrology.flood_early_warning import FloodEarlyWarningSystem
            ews = FloodEarlyWarningSystem()
            ews.push_alert(
                river_id=w["river_id"],
                level=w["level"],
                forecast_cms_6hr=w["forecast_cms_6hr"],
                forecast_cms_12hr=w["forecast_cms_12hr"],
                forecast_cms_24hr=w["forecast_cms_24hr"],
                confidence_pct=w["confidence_pct"],
                expected_peak_time=w.get("expected_peak_time"),
            )
            Variable.set(var_key, str(now_ts))
            dispatched.append(w["river_id"])
            logger.warning("BPBD alert dispatched | %s level=%s", w["river_id"], w["level"])
        except Exception as exc:
            logger.error("Alert dispatch failed for %s: %s", w["river_id"], exc)

    return {"dispatched": dispatched}


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="hydrologis_flood_early_warning",
    description="Every-30-min flood early warning: 6 rivers, BNPB levels, BPBD alert, "
                "active_warnings.json sidecar.",
    schedule_interval="*/30 * * * *",
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    tags=["hydrologis", "flood", "early_warning", "sprint9"],
    doc_md=__doc__,
) as dag:

    t_qpe = ShortCircuitOperator(
        task_id="load_qpe_latest",
        python_callable=load_qpe_latest,
        sla=timedelta(minutes=1),
    )

    t_sf = ShortCircuitOperator(
        task_id="load_streamflow_latest",
        python_callable=load_streamflow_latest,
        sla=timedelta(minutes=1),
    )

    with TaskGroup("evaluate_all_rivers",
                   tooltip="Parallel flood evaluation per river") as tg_eval:
        for _river in _RIVERS:
            PythonOperator(
                task_id=f"evaluate_{_river}",
                python_callable=_make_evaluate_callable(_river),
                sla=timedelta(minutes=3),
            )

    t_write = PythonOperator(
        task_id="write_active_warnings",
        python_callable=write_active_warnings,
        sla=timedelta(minutes=4),
    )

    t_metrics = PythonOperator(
        task_id="emit_warning_metrics",
        python_callable=emit_warning_metrics,
        sla=timedelta(minutes=4, seconds=30),
    )

    t_alert = PythonOperator(
        task_id="dispatch_alerts",
        python_callable=dispatch_alerts,
        sla=timedelta(minutes=5),
    )

    [t_qpe, t_sf] >> tg_eval >> t_write >> t_metrics >> t_alert
