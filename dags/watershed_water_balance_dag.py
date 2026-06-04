"""
watershed_water_balance_dag.py — Sprint 7 G2
dag_id: hydrologis_watershed_water_balance
schedule: 0 2 1 * * Asia/Jakarta (monthly, 1st of month at 02:00 WIB)
SLA: 60 minutes

Tasks:
  load_monthly_inputs   — validate QPE monthly data; skip if absent or > 35 days old
  compute_water_balance — WatershedWaterBalance.compute() for all 20 DAS
  flag_residuals        — log UNCLOSED watersheds and emit Prometheus metrics
  write_balance_reports — confirm per-DAS JSON output files
  write_visualia_summary — generate water_balance_summary_{YYYYMM}.md for VISUALIA

On completion: record_ingestion_success('watershed_water_balance_monthly')
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WORKSPACE  = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
DATA_DIR   = WORKSPACE / "data"
REPORT_DIR = WORKSPACE / "output" / "water_balance"

DAS_IDS = [
    "DAS-CI", "DAS-BR", "DAS-SL", "DAS-MK", "DAS-KP", "DAS-ML", "DAS-SN",
    "DAS-MN", "DAS-PW", "DAS-JR", "DAS-SR", "DAS-CR", "DAS-CM", "DAS-CJ",
    "DAS-BI", "DAS-WK", "DAS-PO", "DAS-TA", "DAS-TM", "DAS-DG",
]

_default_args = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def load_monthly_inputs(**context) -> dict:
    """
    Validate QPE monthly file exists and is not older than 35 days.
    Skips downstream processing if data is absent or stale.
    """
    from airflow.exceptions import AirflowSkipException

    # Determine current month (execution_date is 1st of month)
    exec_date  = context.get("logical_date") or context.get("execution_date")
    month_obj  = exec_date.date() if hasattr(exec_date, "date") else datetime.now(timezone.utc).date()
    month_obj  = month_obj.replace(day=1)
    month_str  = month_obj.strftime("%Y%m")

    qpe_path   = DATA_DIR / f"qpe_monthly_{month_str}.json"
    if not qpe_path.exists():
        raise AirflowSkipException(
            f"QPE monthly file not found at {qpe_path}. "
            "Skipping water balance run."
        )

    # Age check (35-day window)
    import os as _os
    mtime = _os.path.getmtime(qpe_path)
    age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400
    if age_days > 35:
        raise AirflowSkipException(
            f"QPE monthly file is {age_days:.1f} days old (max 35). "
            "Skipping water balance run."
        )

    logger.info("QPE monthly inputs OK | month=%s age=%.1f days", month_str, age_days)

    # Also check GRACE output (informational; not a skip condition)
    grace_path = WORKSPACE / "output" / "aquifer" / f"grace_depletion_{month_str}.json"
    grace_ready = grace_path.exists()
    logger.info("GRACE-FO depletion output available: %s", grace_ready)

    context["ti"].xcom_push(key="month_str",    value=month_str)
    context["ti"].xcom_push(key="grace_ready",  value=grace_ready)
    return {"month_str": month_str, "grace_ready": grace_ready, "qpe_age_days": round(age_days, 1)}


def compute_water_balance(**context) -> dict:
    """
    Run WatershedWaterBalance.compute() for all 20 DAS.
    Pushes per-DAS result summaries via XCom.
    """
    from src.hydrology.watershed_water_balance import WatershedWaterBalance

    ti         = context["ti"]
    month_str  = ti.xcom_pull(task_ids="load_monthly_inputs", key="month_str")
    if not month_str:
        month_str = datetime.now(timezone.utc).strftime("%Y%m")

    month_date = date(int(month_str[:4]), int(month_str[4:6]), 1)
    computer   = WatershedWaterBalance()
    results    = {}

    for das_id in DAS_IDS:
        try:
            result = computer.compute(watershed_id=das_id, month=month_date)
            results[das_id] = {
                "status":             result.status,
                "P_mm":               result.P_mm,
                "ET_mm":              result.ET_mm,
                "Q_mm":               result.Q_mm,
                "delta_S_mm":         result.delta_S_mm,
                "residual_mm":        result.residual_mm,
                "residual_fraction":  result.residual_fraction,
                "et_source":          result.et_source,
                "ds_source":          result.ds_source,
                "output_path":        result.output_path,
                "name":               result.watershed_name,
                "warnings":           result.warnings,
            }
            logger.info(
                "WB | %-8s %-22s P=%6.1f ET=%5.1f Q=%5.1f ΔS=%6.1f res=%6.1f mm %s",
                das_id, result.watershed_name,
                result.P_mm, result.ET_mm, result.Q_mm, result.delta_S_mm,
                result.residual_mm, result.status,
            )
        except Exception as exc:
            logger.error("WB FAILED | %s: %s", das_id, exc, exc_info=True)
            results[das_id] = {"status": "failed", "error": str(exc)}

    ti.xcom_push(key="wb_results",  value=results)
    ti.xcom_push(key="month_str",   value=month_str)
    return results


def flag_residuals(**context) -> dict:
    """
    Log UNCLOSED watersheds and summarise closure statistics.
    Prometheus metrics already emitted inside WatershedWaterBalance.compute().
    """
    ti      = context["ti"]
    results = ti.xcom_pull(task_ids="compute_water_balance", key="wb_results")
    if not results:
        logger.warning("No water balance results in XCom; skipping flag step")
        return {}

    unclosed  = [k for k, v in results.items() if v.get("status") == "UNCLOSED"]
    closed    = [k for k, v in results.items() if v.get("status") == "CLOSED"]
    failed    = [k for k, v in results.items() if v.get("status") == "failed"]

    logger.info("=== Water Balance Closure Summary ===")
    logger.info("  CLOSED:   %2d  UNCLOSED: %2d  FAILED: %2d",
                len(closed), len(unclosed), len(failed))

    if unclosed:
        logger.warning("UNCLOSED watersheds (|res/P| > 15%%): %s", unclosed)
        for das_id in unclosed:
            d = results[das_id]
            logger.warning(
                "  %-8s res=%.1f mm (%.1f%% of P) P=%.1f ET=%.1f Q=%.1f ΔS=%.1f",
                das_id,
                d.get("residual_mm", 0.0),
                d.get("residual_fraction", 0.0) * 100,
                d.get("P_mm", 0.0),
                d.get("ET_mm", 0.0),
                d.get("Q_mm", 0.0),
                d.get("delta_S_mm", 0.0),
            )

    return {
        "unclosed_count": len(unclosed),
        "closed_count":   len(closed),
        "failed_count":   len(failed),
        "unclosed_ids":   unclosed,
    }


def write_balance_reports(**context) -> list[str]:
    """
    Confirm per-DAS balance JSON output files and record ingestion success.
    """
    ti      = context["ti"]
    results = ti.xcom_pull(task_ids="compute_water_balance", key="wb_results")
    if not results:
        return []

    confirmed = []
    for das_id, data in results.items():
        out = data.get("output_path")
        if out and Path(out).exists():
            confirmed.append(out)
        else:
            logger.warning("Output NOT found | %s path=%s", das_id, out)

    logger.info("Balance reports confirmed: %d / %d", len(confirmed), len(DAS_IDS))

    # Record ingestion success
    try:
        from src.hydrology.metrics import record_ingestion_success
        record_ingestion_success("watershed_water_balance_monthly")
    except ImportError:
        try:
            import time
            from src.hydrology.metrics import INGESTION_SUCCESS_TS
            INGESTION_SUCCESS_TS.labels(source="watershed_water_balance_monthly").set(time.time())
        except Exception as exc:
            logger.debug("INGESTION_SUCCESS_TS unavailable: %s", exc)

    return confirmed


def write_visualia_summary(**context) -> str:
    """
    Generate Markdown water balance summary for VISUALIA consumption.
    Output: workspace/output/water_balance/water_balance_summary_{YYYYMM}.md
    """
    ti         = context["ti"]
    results    = ti.xcom_pull(task_ids="compute_water_balance", key="wb_results")
    month_str  = ti.xcom_pull(task_ids="compute_water_balance", key="month_str")

    if not results or not month_str:
        logger.warning("No data for VISUALIA summary; skipping")
        return "no_report"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    run_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    month_fmt = f"{month_str[:4]}-{month_str[4:6]}"
    out_path  = REPORT_DIR / f"water_balance_summary_{month_str}.md"

    # Table rows
    rows = ""
    for das_id in DAS_IDS:
        d = results.get(das_id, {})
        if d.get("status") == "failed":
            rows += f"| {das_id} | {d.get('name', '—')} | — | — | — | — | — | ❌ FAILED |\n"
            continue
        emoji  = "✅" if d.get("status") == "CLOSED" else "⚠️"
        rows += (
            f"| {das_id} | {d.get('name','—')} "
            f"| {d.get('P_mm',0):.1f} | {d.get('ET_mm',0):.1f} "
            f"| {d.get('Q_mm',0):.1f} | {d.get('delta_S_mm',0):.1f} "
            f"| {d.get('residual_mm',0):.1f} ({d.get('residual_fraction',0)*100:.1f}%) "
            f"| {emoji} {d.get('status','—')} |\n"
        )

    unclosed  = [k for k, v in results.items() if v.get("status") == "UNCLOSED"]
    closed_ct = sum(1 for v in results.values() if v.get("status") == "CLOSED")
    failed_ct = sum(1 for v in results.values() if v.get("status") == "failed")

    unclosed_sec = ""
    if unclosed:
        unclosed_sec = "\n## ⚠️ UNCLOSED Watersheds (|residual/P| > 15%)\n\n"
        for das_id in unclosed:
            d = results[das_id]
            unclosed_sec += (
                f"- **{d.get('name', das_id)}** ({das_id}): "
                f"residual = {d.get('residual_mm', 0):.1f} mm "
                f"({d.get('residual_fraction', 0)*100:.1f}% of P)\n"
            )

    report = f"""# Water Balance Summary — {month_fmt}

*Generated by HYDROLOGIS Sprint 7 | {run_ts}*

## Summary

| Metric | Value |
|--------|-------|
| Month | {month_fmt} |
| DAS Assessed | {len(DAS_IDS)} |
| CLOSED | {closed_ct} |
| UNCLOSED | {len(unclosed)} |
| FAILED | {failed_ct} |
| Closure criterion | \|residual/P\| ≤ 15% |
{unclosed_sec}
## Full DAS Water Balance Table

| DAS ID | Name | P (mm) | ET (mm) | Q (mm) | ΔS (mm) | Residual | Status |
|--------|------|--------|---------|--------|---------|----------|--------|
{rows.rstrip()}

## Component Notes

- **P**: QPE monthly accumulation
- **ET**: MODIS ET product (or Penman-Monteith fallback)
- **Q**: Streamflow monthly integral (rational method)
- **ΔS**: SMAP root-zone + GRACE-FO TWS anomaly
- **Residual** = P − ET − Q − ΔS

---
*Consumed by VISUALIA dashboard. Next run: 1st of following month at 02:00 WIB.*
"""

    with open(out_path, "w") as f:
        f.write(report)

    logger.info("VISUALIA water balance summary written | path=%s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id            = "hydrologis_watershed_water_balance",
    description       = "Sprint 7: Monthly water balance closure for 20 DAS Strategis Nasional",
    default_args      = _default_args,
    schedule_interval = "0 2 1 * *",  # 1st of month at 02:00 (server UTC; WIB offset handled by Airflow timezone)
    start_date        = days_ago(32),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["hydrologis", "water_balance", "sprint7"],
    dagrun_timeout    = timedelta(minutes=60),
) as dag:

    t_inputs  = PythonOperator(task_id="load_monthly_inputs",   python_callable=load_monthly_inputs)
    t_compute = PythonOperator(task_id="compute_water_balance",  python_callable=compute_water_balance)
    t_flag    = PythonOperator(task_id="flag_residuals",         python_callable=flag_residuals)
    t_reports = PythonOperator(task_id="write_balance_reports",  python_callable=write_balance_reports)
    t_vis     = PythonOperator(task_id="write_visualia_summary", python_callable=write_visualia_summary)

    t_inputs >> t_compute >> t_flag >> t_reports >> t_vis
