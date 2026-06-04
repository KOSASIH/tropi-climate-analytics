"""
drought_risk_dag.py — Sprint 6 E4
dag_id: hydrologis_drought_risk
schedule: 0 */6 * * * Asia/Jakarta (every 6 hours)
SLA: 15 minutes

Tasks:
  load_smap_data      — load workspace/data/smap_latest.json (or note absence)
  compute_drought_risk — DroughtRiskMonitor.run(assessment_date)
  log_drought_metrics  — log per-DAS risk class summary
  write_drought_report — generate workspace/output/drought/drought_summary_{YYYYMMDD}.md → VISUALIA
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WORKSPACE  = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
SMAP_PATH  = WORKSPACE / "data" / "smap_latest.json"
REPORT_DIR = WORKSPACE / "output" / "drought"

_default_args = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def load_smap_data(**context) -> dict:
    """
    Load SMAP soil moisture file. Records whether SMAP is present or if
    the run will fall back to climatological mean values.
    Does NOT skip — DroughtRiskMonitor handles the fallback gracefully.
    """
    if SMAP_PATH.exists():
        try:
            with open(SMAP_PATH) as f:
                smap = json.load(f)
            das_count = len(smap) if isinstance(smap, dict) else 0
            logger.info("SMAP loaded | path=%s das_count=%d", SMAP_PATH, das_count)
            return {"smap_available": True, "das_count": das_count, "path": str(SMAP_PATH)}
        except json.JSONDecodeError as exc:
            logger.warning("SMAP file parse error: %s — will use climatological fallback", exc)
            return {"smap_available": False, "reason": f"parse_error: {exc}"}
    else:
        logger.info("SMAP file absent at %s — will use climatological fallback", SMAP_PATH)
        return {"smap_available": False, "reason": "file_not_found"}


def compute_drought_risk(**context) -> dict:
    """
    Run DroughtRiskMonitor for today's assessment date.
    Returns serialisable summary via XCom.
    """
    from src.hydrology.drought_risk_monitor import DroughtRiskMonitor

    assessment_date = datetime.now(timezone.utc).date()
    monitor  = DroughtRiskMonitor()
    result   = monitor.run(assessment_date=assessment_date)

    summary = {
        "assessment_date":  result.assessment_date,
        "status":           result.status,
        "warning_count":    result.warning_count,
        "emergency_count":  result.emergency_count,
        "imbalanced":       result.imbalanced_watersheds,
        "output_path":      result.output_path,
        "notes":            result.notes,
        "watersheds": [
            {
                "id":         w.watershed_id,
                "name":       w.watershed_name,
                "risk_class": w.risk_class,
                "risk_label": w.risk_label,
                "swd_pct":    w.swd_pct,
                "source":     w.source,
            }
            for w in result.watersheds
        ],
    }

    logger.info(
        "Drought risk | date=%s status=%s warning=%d emergency=%d",
        result.assessment_date, result.status,
        result.warning_count, result.emergency_count,
    )
    if result.imbalanced_watersheds:
        logger.warning(
            "WARNING/EMERGENCY watersheds: %s", result.imbalanced_watersheds
        )

    context["ti"].xcom_push(key="drought_summary", value=summary)
    return summary


def log_drought_metrics(**context) -> dict:
    """
    Log per-DAS risk class summary table to Airflow task log for observability.
    Prometheus metrics are already emitted inside DroughtRiskMonitor.run().
    """
    ti      = context["ti"]
    summary = ti.xcom_pull(task_ids="compute_drought_risk", key="drought_summary")
    if not summary:
        logger.warning("No drought summary in XCom; skipping metrics log")
        return {}

    logger.info(
        "=== Drought Risk Summary | %s ===", summary.get("assessment_date", "N/A")
    )
    logger.info(
        "  Total warning (≥WATCH): %-3d  Emergency: %-3d  Status: %s",
        summary.get("warning_count", 0),
        summary.get("emergency_count", 0),
        summary.get("status", "unknown"),
    )
    logger.info("  %-10s %-22s %-5s %-12s %s",
                "ID", "Name", "Class", "Label", "SWD%")
    logger.info("  " + "-" * 65)
    for w in summary.get("watersheds", []):
        logger.info(
            "  %-10s %-22s %-5d %-12s %.1f%%",
            w["id"], w["name"], w["risk_class"], w["risk_label"], w["swd_pct"],
        )

    if summary.get("notes"):
        for note in summary["notes"]:
            logger.warning("  NOTE: %s", note)

    return {
        "warning_count":   summary.get("warning_count", 0),
        "emergency_count": summary.get("emergency_count", 0),
    }


def write_drought_report(**context) -> str:
    """
    Generate a Markdown drought summary for VISUALIA consumption.
    Output: workspace/output/drought/drought_summary_{YYYYMMDD}.md
    """
    ti      = context["ti"]
    summary = ti.xcom_pull(task_ids="compute_drought_risk", key="drought_summary")
    if not summary:
        logger.warning("No drought summary in XCom; skipping report")
        return "no_report"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    date_str  = summary.get("assessment_date", datetime.now(timezone.utc).date().isoformat())
    date_tag  = date_str.replace("-", "")
    run_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out_path  = REPORT_DIR / f"drought_summary_{date_tag}.md"

    # Build risk class table
    risk_rows = ""
    for w in summary.get("watersheds", []):
        emoji = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}.get(w["risk_class"], "⚪")
        risk_rows += (
            f"| {w['id']} | {w['name']} | {w['swd_pct']:.1f}% "
            f"| {emoji} {w['risk_label']} | {w['source']} |\n"
        )

    # Highlight section for elevated watersheds
    elevated = [w for w in summary.get("watersheds", []) if w["risk_class"] >= 2]
    elevated_section = ""
    if elevated:
        elevated_section = "\n## ⚠️ Elevated Risk Watersheds (≥WARNING)\n\n"
        for w in elevated:
            emoji = "🟠" if w["risk_class"] == 2 else "🔴"
            elevated_section += (
                f"- **{w['name']}** ({w['id']}): {emoji} {w['risk_label']} "
                f"— SWD {w['swd_pct']:.1f}%\n"
            )

    notes_section = ""
    if summary.get("notes"):
        notes_section = "\n## Data Notes\n\n"
        for note in summary["notes"]:
            notes_section += f"> ⚠️ {note}\n"

    report = f"""# Drought Risk Assessment — {date_str}

*Generated by HYDROLOGIS Sprint 6 | {run_ts}*

## Summary

| Metric | Value |
|--------|-------|
| Assessment Date | {date_str} |
| Watersheds Assessed | {len(summary.get('watersheds', []))} |
| At-Risk (≥WATCH) | {summary.get('warning_count', 0)} |
| Emergency | {summary.get('emergency_count', 0)} |
| Data Source | {'SMAP' if summary.get('status') == 'ok' else 'Climatological fallback'} |
| Run Status | {summary.get('status', 'unknown').upper()} |
{elevated_section}
## Full DAS Risk Table

| DAS ID | Watershed | SWD % | Risk Class | Source |
|--------|-----------|-------|------------|--------|
{risk_rows.rstrip()}

## Risk Class Legend

| Class | Label | SWD Range | Action |
|-------|-------|-----------|--------|
| 0 | 🟢 NORMAL    | < 20%      | Routine monitoring |
| 1 | 🟡 WATCH     | 20%–40%    | Increased monitoring |
| 2 | 🟠 WARNING   | 40%–60%    | Advisory to agriculture sector |
| 3 | 🔴 EMERGENCY | ≥ 60%      | Immediate intervention required |
{notes_section}
---
*Report consumed by VISUALIA dashboard. Next update in ~6 hours.*
"""

    with open(out_path, "w") as f:
        f.write(report)

    logger.info("Drought summary report written | path=%s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id             = "hydrologis_drought_risk",
    description        = "Sprint 6: SMAP-based drought risk assessment for 20 DAS — every 6 hours",
    default_args       = _default_args,
    schedule_interval  = "0 */6 * * *",
    start_date         = days_ago(1),
    catchup            = False,
    max_active_runs    = 1,
    tags               = ["hydrologis", "drought", "sprint6"],
    dagrun_timeout     = timedelta(minutes=15),
) as dag:

    t_smap = PythonOperator(
        task_id         = "load_smap_data",
        python_callable = load_smap_data,
    )

    t_risk = PythonOperator(
        task_id         = "compute_drought_risk",
        python_callable = compute_drought_risk,
    )

    t_metrics = PythonOperator(
        task_id         = "log_drought_metrics",
        python_callable = log_drought_metrics,
    )

    t_report = PythonOperator(
        task_id         = "write_drought_report",
        python_callable = write_drought_report,
    )

    t_smap >> t_risk >> t_metrics >> t_report
