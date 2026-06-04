"""
Airflow DAG — Agricultural Water Advisory (Weekly)
HYDROLOGIS Sprint 4 | Deliverable 3

dag_id:   hydrologis_agri_advisory
schedule: 0 7 * * 1 (Asia/Jakarta) — Monday 07:00 WIB
SLA:      20 minutes

Task graph:
  prepare_inputs → generate_all_advisories → log_advisory_metrics → write_summary_report
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("WORKSPACE_ROOT", "workspace")

default_args = {
    "owner":             "hydrologis",
    "depends_on_past":   False,
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=20),
}

# ---------------------------------------------------------------------------
# Task 1: prepare_inputs
# ---------------------------------------------------------------------------

def prepare_inputs(**context) -> dict:
    """
    Resolve forecast_month from logical_date and verify Sprint 2 WAI outputs exist.
    Sprint 2 seasonal output path: workspace/output/seasonal/water_availability_advisory_YYYYMM.json
    """
    logical_date = context["logical_date"]
    # Advisory is for the CURRENT month (issued on first Monday of month)
    forecast_month = logical_date.replace(day=1)

    seasonal_dir = os.path.join(WORKSPACE, "output", "seasonal")
    os.makedirs(seasonal_dir, exist_ok=True)

    wai_fname = f"water_availability_advisory_{forecast_month.strftime('%Y%m')}.json"
    wai_path  = os.path.join(seasonal_dir, wai_fname)

    if os.path.exists(wai_path):
        logger.info("Sprint 2 WAI output found: %s", wai_path)
        with open(wai_path) as fh:
            wai_meta = json.load(fh)
        enso_phase = wai_meta.get("enso_phase", "NEUTRAL")
    else:
        logger.warning("Sprint 2 WAI output missing at %s — proceeding with NEUTRAL ENSO fallback", wai_path)
        enso_phase = "NEUTRAL"

    return {
        "forecast_month": forecast_month.isoformat(),
        "wai_path_exists": os.path.exists(wai_path),
        "enso_phase": enso_phase,
    }


# ---------------------------------------------------------------------------
# Task 2: generate_all_advisories
# ---------------------------------------------------------------------------

def generate_all_advisories(**context) -> dict:
    """
    Run AgriWaterAdvisoryGenerator for all 20 strategic watersheds.
    Generates advisories for 30, 60, and 90-day horizons.
    """
    from src.hydrology.agri_water_advisory import AgriWaterAdvisoryGenerator

    ti = context["ti"]
    inputs = ti.xcom_pull(task_ids="prepare_inputs")
    forecast_month = date.fromisoformat(inputs["forecast_month"])

    gen = AgriWaterAdvisoryGenerator()
    status = gen.run(
        forecast_month=forecast_month,
        advisory_horizons=[30, 60, 90],
    )

    logger.info(
        "Agri advisory complete | month=%s watersheds=%d critical=%d deficit=%d",
        forecast_month,
        status.watersheds_processed,
        len(status.watersheds_critical),
        len(status.watersheds_deficit),
    )

    return {
        "forecast_month":       forecast_month.isoformat(),
        "enso_phase":           status.enso_phase,
        "watersheds_processed": status.watersheds_processed,
        "watersheds_critical":  status.watersheds_critical,
        "watersheds_deficit":   status.watersheds_deficit,
        "output_path":          status.output_path,
        "success":              status.success,
    }


# ---------------------------------------------------------------------------
# Task 3: log_advisory_metrics
# ---------------------------------------------------------------------------

def log_advisory_metrics(**context) -> None:
    """
    Push tropi_agri_water_advisory_class metrics to Prometheus Pushgateway.
    Metrics were already set during generate_all_advisories via _record_advisory_class().
    This task confirms push and logs a summary.
    """
    from src.hydrology.metrics import push_metrics

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="generate_all_advisories")

    # Confirm push
    push_metrics()

    n_critical = len(result.get("watersheds_critical", []))
    n_deficit  = len(result.get("watersheds_deficit", []))

    logger.info(
        "Advisory metrics pushed | ENSO=%s critical=%d deficit=%d",
        result.get("enso_phase"),
        n_critical,
        n_deficit,
    )

    if n_critical > 0:
        logger.warning(
            "CRITICAL water availability in %d watersheds: %s",
            n_critical,
            result.get("watersheds_critical"),
        )


# ---------------------------------------------------------------------------
# Task 4: write_summary_report
# ---------------------------------------------------------------------------

def write_summary_report(**context) -> str:
    """
    Write a Markdown weekly advisory summary to workspace/output/agri_advisory/summary_{YYYYMM}.md.
    This report is picked up by VISUALIA for dashboard rendering.
    """
    ti = context["ti"]
    inputs = ti.xcom_pull(task_ids="prepare_inputs")
    result = ti.xcom_pull(task_ids="generate_all_advisories")

    forecast_month = date.fromisoformat(result["forecast_month"])
    summary_dir  = os.path.join(WORKSPACE, "output", "agri_advisory")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"summary_{forecast_month.strftime('%Y%m')}.md")

    critical_list = "\n".join(
        f"  - {ws}" for ws in result.get("watersheds_critical", [])
    ) or "  _(none)_"
    deficit_list = "\n".join(
        f"  - {ws}" for ws in result.get("watersheds_deficit", [])
    ) or "  _(none)_"

    md = f"""# Ringkasan Prakiraan Air Irigasi — {forecast_month.strftime('%B %Y')}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**ENSO Phase:** {result.get('enso_phase', 'NEUTRAL')}  
**Watersheds Processed:** {result.get('watersheds_processed', 0)}  

## Status Ketersediaan Air

### 🔴 CRITICAL ({len(result.get('watersheds_critical', []))})
{critical_list}

### 🟡 DEFICIT ({len(result.get('watersheds_deficit', []))})
{deficit_list}

## Rekomendasi Singkat

{"⚠️ **SIAGA AIR**: Ada DAS dalam kondisi kritis — segera koordinasi BPSDA untuk alokasi darurat." if result.get('watersheds_critical') else "✅ Tidak ada DAS dalam kondisi kritis minggu ini."}

## Output Files
- Advisory JSON: `{result.get('output_path', 'N/A')}`
- WAI Source: `workspace/output/seasonal/water_availability_advisory_{forecast_month.strftime('%Y%m')}.json`

---
_HYDROLOGIS Sprint 4 | Tropi Climate Analytics_
"""

    with open(summary_path, "w") as fh:
        fh.write(md)
    logger.info("Agri advisory summary written: %s", summary_path)
    return summary_path



# ---------------------------------------------------------------------------
# Task 5: write_water_balance_report  (Sprint 5 addition)
# ---------------------------------------------------------------------------

def write_water_balance_report(**context) -> str:
    """
    Invoke WaterBalanceReportGenerator for current forecast month.
    Writes workspace/output/reports/water_balance_YYYYMM.md for VISUALIA.
    """
    from src.hydrology.water_balance_report import WaterBalanceReportGenerator

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="generate_all_advisories")
    forecast_month = date.fromisoformat(result["forecast_month"])

    gen = WaterBalanceReportGenerator()
    report_path = gen.run(forecast_month=forecast_month)
    logger.info("Water balance report written: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="hydrologis_agri_advisory",
    description="Weekly agricultural water availability advisory for 20 DAS Strategis Nasional",
    schedule="0 7 * * 1",
    start_date=datetime(2026, 1, 6),    # First Monday of 2026
    catchup=False,
    default_args=default_args,
    tags=["hydrologis", "agri", "irrigation", "weekly"],
    doc_md="""
## HYDROLOGIS — Agricultural Water Advisory (Weekly)

**Schedule:** Every Monday at 07:00 WIB (Asia/Jakarta)  
**SLA:** 20 minutes  
**Scope:** 20 DAS Strategis Nasional (Kep. MenLHK P.10/2019)  
**Horizons:** 30 / 60 / 90 days  

**Inputs:**  
- Sprint 2 seasonal WAI output (`workspace/output/seasonal/`)  
- SMAP root-zone soil moisture (stub → production Sprint 5)  
- BMKG ENSO seasonal outlook  

**Prometheus:** `tropi_agri_water_advisory_class{watershed_id}` (0=SURPLUS … 3=CRITICAL)  
**Summary report:** `workspace/output/agri_advisory/summary_YYYYMM.md` → VISUALIA
    """,
    dagrun_timeout=timedelta(minutes=20),
) as dag:

    t_prepare = PythonOperator(
        task_id="prepare_inputs",
        python_callable=prepare_inputs,
    )

    t_generate = PythonOperator(
        task_id="generate_all_advisories",
        python_callable=generate_all_advisories,
    )

    t_metrics = PythonOperator(
        task_id="log_advisory_metrics",
        python_callable=log_advisory_metrics,
    )

    t_report = PythonOperator(
        task_id="write_summary_report",
        python_callable=write_summary_report,
    )

    t_water_balance = PythonOperator(
        task_id="write_water_balance_report",
        python_callable=write_water_balance_report,
    )

    # Task dependencies
    t_prepare >> t_generate >> t_metrics >> t_report >> t_water_balance


# ---------------------------------------------------------------------------
# Task 5: write_water_balance_report  (Sprint 5 D5 integration)
# ---------------------------------------------------------------------------

def write_water_balance_report(**context) -> str:
    """
    Generate monthly P−ET−ΔS=Q water balance report for all 20 DAS.
    Writes workspace/output/reports/water_balance_YYYYMM.md for VISUALIA.
    Flags watersheds with closure error > 15%.
    """
    from datetime import date as date_cls

    from src.hydrology.water_balance_report import WaterBalanceReportGenerator

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="generate_all_advisories")

    # Report month matches the advisory month
    forecast_month = date_cls.fromisoformat(result["forecast_month"])

    gen    = WaterBalanceReportGenerator()
    status = gen.run(report_month=forecast_month)

    logger.info(
        "Water balance report written | month=%s imbalanced=%d path=%s",
        forecast_month,
        len(status.imbalanced_watersheds),
        status.report_path,
    )

    if status.imbalanced_watersheds:
        logger.warning(
            "Water balance imbalance >15%% in: %s",
            status.imbalanced_watersheds,
        )

    return status.report_path
