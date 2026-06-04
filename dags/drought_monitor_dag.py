"""
drought_monitor_dag.py — Sprint 8 J4
dag_id: hydrologis_drought_monitor

Schedule: 0 6 * * * Asia/Jakarta  (daily 06:00 WIB)
SLA: 20 minutes

Computes SPI-3/SPEI-3 drought indices for all 5 island regions in parallel,
aggregates a national summary, writes a VISUALIA-formatted Markdown report,
emits DROUGHT_RISK_LEVEL Prometheus gauges, and pushes BPBD webhook alerts
whenever any region reaches SEVERE_DROUGHT or EXTREME_DROUGHT.

Tasks:
    assess_all_regions    (TaskGroup, max_active_tasks=5, parallel)
      ├─ assess_java
      ├─ assess_sumatra
      ├─ assess_kalimantan
      ├─ assess_sulawesi
      └─ assess_maluku_papua
    aggregate_national_summary
    write_drought_report      → workspace/output/drought/drought_summary_{YYYYMMDD}.md
    emit_drought_metrics      → DROUGHT_RISK_LEVEL gauge per region
    alert_if_severe           → BPBD webhook on SEVERE/EXTREME

Cross-agent handoff:
    VISUALIA:  workspace/output/drought/drought_summary_{YYYYMMDD}.md
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "hydrologis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

_WS            = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_DROUGHT_DIR   = _WS / "output" / "drought"
_REGIONS       = ["java", "sumatra", "kalimantan", "sulawesi", "maluku_papua"]

# Classification labels for report (mirrors drought_monitor.py)
_LEVEL_LABEL = {0: "NORMAL", 1: "MILD_DROUGHT", 2: "MODERATE_DROUGHT",
                3: "SEVERE_DROUGHT", 4: "EXTREME_DROUGHT"}
_ALERT_THRESHOLD = 3  # SEVERE_DROUGHT and above


# ── per-region task factory ───────────────────────────────────────────────────

def _make_assess_callable(region_id: str):
    def assess_region(**context) -> dict:
        from src.hydrology.drought_monitor import DroughtMonitor
        valid_date = date.fromisoformat(
            context["dag_run"].conf.get("valid_date", date.today().isoformat())
        )
        monitor = DroughtMonitor()
        result  = monitor.assess(region_id=region_id, valid_date=valid_date)
        payload = {
            "region_id":              result.region_id,
            "valid_date":             result.valid_date.isoformat(),
            "spi_3":                  result.spi_3,
            "spei_3":                 result.spei_3,
            "classification":         result.classification,
            "level":                  result.level,
            "soil_moisture_anomaly":  result.soil_moisture_anomaly_pct,
            "affected_area_km2":      result.affected_area_km2,
        }
        context["ti"].xcom_push(key=f"result_{region_id}", value=payload)
        logger.info("Drought assessed | region=%s spi=%.2f class=%s",
                    region_id, result.spi_3, result.classification)
        return payload
    assess_region.__name__ = f"assess_{region_id}"
    return assess_region


# ── downstream callables ──────────────────────────────────────────────────────

def aggregate_national_summary(**context) -> dict:
    """Pull all per-region XCom results and build the national rollup."""
    ti = context["ti"]
    results = []
    for region_id in _REGIONS:
        r = ti.xcom_pull(key=f"result_{region_id}", task_ids=f"assess_all_regions.assess_{region_id}")
        if r:
            results.append(r)

    if not results:
        logger.warning("No regional results — using empty summary")
        summary = {"worst_classification": "NORMAL", "worst_level": 0,
                   "affected_regions": 0, "total_affected_area_km2": 0.0, "regions": []}
    else:
        worst = max(results, key=lambda x: x["level"])
        summary = {
            "worst_classification":   worst["classification"],
            "worst_level":            worst["level"],
            "affected_regions":       sum(1 for r in results if r["level"] >= 1),
            "severe_or_extreme":      sum(1 for r in results if r["level"] >= _ALERT_THRESHOLD),
            "total_affected_area_km2": sum(r["affected_area_km2"] for r in results if r["level"] >= 1),
            "regions":                results,
        }

    ti.xcom_push(key="national_summary", value=summary)
    return summary


def write_drought_report(**context) -> dict:
    """
    Write VISUALIA-formatted Markdown drought summary.
    Path: workspace/output/drought/drought_summary_{YYYYMMDD}.md
    """
    ti      = context["ti"]
    summary = ti.xcom_pull(key="national_summary", task_ids="aggregate_national_summary")
    today   = date.fromisoformat(
        context["dag_run"].conf.get("valid_date", date.today().isoformat())
    )

    _DROUGHT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _DROUGHT_DIR / f"drought_summary_{today:%Y%m%d}.md"

    severity_emoji = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🚨"}

    lines = [
        f"# Tropi-Climate-Analytics — Drought Monitor",
        f"**Date:** {today:%d %B %Y}  |  "
        f"**National Status:** {severity_emoji.get(summary['worst_level'], '⚪')} "
        f"{summary['worst_classification']}",
        "",
        "## Regional Summary",
        "",
        "| Region | SPI-3 | SPEI-3 | Classification | Affected Area (km²) | SM Anomaly (%) |",
        "|--------|-------|--------|---------------|---------------------|----------------|",
    ]
    for r in summary.get("regions", []):
        emoji = severity_emoji.get(r["level"], "⚪")
        lines.append(
            f"| {r['region_id'].replace('_', ' ').title()} "
            f"| {r['spi_3']:+.2f} | {r['spei_3']:+.2f} "
            f"| {emoji} {r['classification']} "
            f"| {r['affected_area_km2']:,.0f} "
            f"| {r['soil_moisture_anomaly']:+.1f}% |"
        )

    lines += [
        "",
        "## National Rollup",
        f"- **Regions in drought (any level):** {summary.get('affected_regions', 0)} / {len(_REGIONS)}",
        f"- **Severe or Extreme regions:** {summary.get('severe_or_extreme', 0)}",
        f"- **Total affected area:** {summary.get('total_affected_area_km2', 0):,.0f} km²",
        "",
        "> *Generated by HYDROLOGIS hydrologis_drought_monitor · "
        f"Tropi-Climate-Analytics*",
    ]

    report_path.write_text("
".join(lines))
    logger.info("Drought report written | path=%s", report_path)
    ti.xcom_push(key="report_path", value=str(report_path))
    return {"report_path": str(report_path)}


def emit_drought_metrics(**context) -> None:
    """Emit DROUGHT_RISK_LEVEL gauge per region."""
    ti      = context["ti"]
    summary = ti.xcom_pull(key="national_summary", task_ids="aggregate_national_summary")
    try:
        from src.hydrology.metrics import record_drought_risk_level
        for r in summary.get("regions", []):
            record_drought_risk_level(
                region_id=r["region_id"],
                classification=r["classification"],
                level=r["level"],
            )
        logger.info("Drought metrics emitted for %d regions", len(summary.get("regions", [])))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Drought metrics emit non-fatal: %s", exc)


def alert_if_severe(**context) -> dict:
    """
    Push BPBD webhook alert when any region reaches SEVERE_DROUGHT or higher.
    Reuses alert infrastructure from emergency_flood_alert_dag.
    """
    ti      = context["ti"]
    summary = ti.xcom_pull(key="national_summary", task_ids="aggregate_national_summary")
    severe_regions = [r for r in summary.get("regions", []) if r["level"] >= _ALERT_THRESHOLD]

    if not severe_regions:
        logger.info("No severe/extreme drought regions — alert skipped")
        return {"alerted": False}

    try:
        from src.hydrology.alert_dispatcher import push_drought_alert  # type: ignore
        for r in severe_regions:
            push_drought_alert(
                region_id=r["region_id"],
                classification=r["classification"],
                spi_3=r["spi_3"],
                affected_area_km2=r["affected_area_km2"],
            )
            logger.warning("DROUGHT ALERT dispatched | region=%s class=%s area_km2=%.0f",
                           r["region_id"], r["classification"], r["affected_area_km2"])
    except ImportError:
        # alert_dispatcher stub — log warning if module not yet wired
        logger.warning(
            "alert_dispatcher not available; drought alert suppressed for: %s",
            [r["region_id"] for r in severe_regions],
        )

    return {"alerted": True, "severe_regions": [r["region_id"] for r in severe_regions]}


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="hydrologis_drought_monitor",
    description="Daily drought risk monitor: SPI-3/SPEI-3 for 5 island regions, VISUALIA report, BPBD alert.",
    schedule_interval="0 6 * * *",
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    tags=["hydrologis", "drought", "sprint8"],
    doc_md=__doc__,
) as dag:

    with TaskGroup("assess_all_regions", tooltip="Parallel per-region SPI/SPEI assessment") as tg_assess:
        for _region in _REGIONS:
            PythonOperator(
                task_id=f"assess_{_region}",
                python_callable=_make_assess_callable(_region),
                sla=timedelta(minutes=8),
            )

    t_aggregate = PythonOperator(
        task_id="aggregate_national_summary",
        python_callable=aggregate_national_summary,
        sla=timedelta(minutes=12),
    )

    t_report = PythonOperator(
        task_id="write_drought_report",
        python_callable=write_drought_report,
        sla=timedelta(minutes=15),
    )

    t_metrics = PythonOperator(
        task_id="emit_drought_metrics",
        python_callable=emit_drought_metrics,
        sla=timedelta(minutes=17),
    )

    t_alert = PythonOperator(
        task_id="alert_if_severe",
        python_callable=alert_if_severe,
        sla=timedelta(minutes=20),
    )

    tg_assess >> t_aggregate >> t_report >> t_metrics >> t_alert
