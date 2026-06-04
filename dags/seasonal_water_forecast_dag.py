"""
seasonal_water_forecast_dag.py — Sprint 9 K4
dag_id: hydrologis_seasonal_water_forecast

Schedule: 0 5 1 * * Asia/Jakarta  (monthly — 1st at 05:00 WIB)
SLA: 30 minutes

Produces 6-month water availability outlook for 5 strategic watersheds using
SeasonalWaterForecaster (harmonic regression + SMAP bias correction + SPEI-3
confidence adjustment). Writes VISUALIA-formatted seasonal_summary_{YYYYMM}.md,
emits SEASONAL_FORECAST_SKILL gauges, verifies prior month 1-month-ahead skill.

Watersheds: citarum, brantas, solo, musi, kapuas

Tasks:
    assess_drought_context       → pull latest SPEI-3 per watershed for bias correction
    forecast_all_watersheds      (TaskGroup, 5 parallel)
      ├─ forecast_citarum
      ├─ forecast_brantas
      ├─ forecast_solo
      ├─ forecast_musi
      └─ forecast_kapuas
    aggregate_national_water_outlook
    write_seasonal_summary       → workspace/output/seasonal_forecast/seasonal_summary_{YYYYMM}.md
    emit_forecast_metrics        → SEASONAL_FORECAST_SKILL per watershed
    verify_prior_month_skill     → Pearson r forecast vs QPE, write skill_scores.json

Cross-agent handoff:
    VISUALIA:   workspace/output/seasonal_forecast/seasonal_summary_{YYYYMM}.md
    ANALYTICA:  workspace/output/seasonal_forecast/forecast_{watershed_id}_{YYYYMM}.json
"""

from __future__ import annotations

import json
import logging
import os
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "hydrologis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

_WS              = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_SF_DIR          = _WS / "output" / "seasonal_forecast"
_DROUGHT_DIR     = _WS / "output" / "drought"
_QPE_DIR         = _WS / "output" / "qpe"
_WATERSHEDS      = ["citarum", "brantas", "solo", "musi", "kapuas"]
_ADVISORY_EMOJI  = {"FAVORABLE": "🟢", "CAUTION": "🟡", "DEFICIT": "🟠", "CRITICAL": "🔴"}


# ── drought context ───────────────────────────────────────────────────────────

def assess_drought_context(**context) -> dict:
    """Load latest SPEI-3 per watershed for SeasonalWaterForecaster bias correction."""
    spei_map: dict = {}
    for ws_id in _WATERSHEDS:
        # Try latest drought assessment JSON written by DroughtMonitor
        pattern = sorted(_DROUGHT_DIR.glob(f"drought_{ws_id}_*.json"))
        if pattern:
            data    = json.loads(pattern[-1].read_text())
            spei_3  = data.get("spei_3", 0.0)
        else:
            spei_3 = 0.0  # neutral if not yet available
        spei_map[ws_id] = spei_3
        logger.info("SPEI-3 context | %s spei=%.2f", ws_id, spei_3)
    context["ti"].xcom_push(key="spei_context", value=spei_map)
    return spei_map


# ── per-watershed forecast factory ───────────────────────────────────────────

def _make_forecast_callable(watershed_id: str):
    def forecast_watershed(**context):
        from src.hydrology.seasonal_water_forecast import SeasonalWaterForecaster
        ti         = context["ti"]
        spei_map   = ti.xcom_pull(key="spei_context", task_ids="assess_drought_context") or {}
        issue_date = date.fromisoformat(
            context["dag_run"].conf.get("issue_date", date.today().isoformat())
        )
        forecaster = SeasonalWaterForecaster()
        result = forecaster.forecast(
            watershed_id=watershed_id,
            issue_date=issue_date,
            horizon_months=6,
            spei_3_override=spei_map.get(watershed_id),
        )
        payload = {
            "watershed_id":         result.watershed_id,
            "issue_date":           result.issue_date.isoformat(),
            "horizon_months":       result.horizon_months,
            "monthly_precip_mm":    result.monthly_precip_mm,
            "monthly_runoff_mm":    result.monthly_runoff_mm,
            "water_deficit_months": result.water_deficit_months,
            "confidence_band_pct":  result.confidence_band_pct,
            "spei_adjusted":        result.spei_adjusted,
            "agricultural_advisory": result.agricultural_advisory,
        }
        # Write per-watershed JSON (ANALYTICA input)
        _SF_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _SF_DIR / f"forecast_{watershed_id}_{issue_date:%Y%m}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        ti.xcom_push(key=f"forecast_{watershed_id}", value=payload)
        logger.info("Seasonal forecast | %s advisory=%s deficit_months=%s",
                    watershed_id, result.agricultural_advisory,
                    sum(result.water_deficit_months))
        return payload
    forecast_watershed.__name__ = f"forecast_{watershed_id}"
    return forecast_watershed


# ── downstream tasks ──────────────────────────────────────────────────────────

def aggregate_national_water_outlook(**context) -> dict:
    ti = context["ti"]
    forecasts = []
    for ws_id in _WATERSHEDS:
        f = ti.xcom_pull(
            key=f"forecast_{ws_id}",
            task_ids=f"forecast_all_watersheds.forecast_{ws_id}",
        )
        if f:
            forecasts.append(f)

    advisory_rank = {"FAVORABLE": 0, "CAUTION": 1, "DEFICIT": 2, "CRITICAL": 3}
    worst_advisory = max(
        (f["agricultural_advisory"] for f in forecasts),
        key=lambda a: advisory_rank.get(a, 0),
        default="FAVORABLE",
    ) if forecasts else "FAVORABLE"

    outlook = {
        "watersheds_assessed":      len(forecasts),
        "worst_advisory":           worst_advisory,
        "deficit_watersheds":       sum(1 for f in forecasts
                                         if f["agricultural_advisory"] in ("DEFICIT", "CRITICAL")),
        "total_deficit_months_avg": sum(sum(f["water_deficit_months"]) for f in forecasts)
                                    / max(len(forecasts), 1),
        "forecasts":                forecasts,
    }
    ti.xcom_push(key="national_outlook", value=outlook)
    return outlook


def write_seasonal_summary(**context) -> dict:
    ti      = context["ti"]
    outlook = ti.xcom_pull(key="national_outlook",
                            task_ids="aggregate_national_water_outlook")
    issue_date = date.fromisoformat(
        context["dag_run"].conf.get("issue_date", date.today().isoformat())
    )
    _SF_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _SF_DIR / f"seasonal_summary_{issue_date:%Y%m}.md"

    worst = outlook.get("worst_advisory", "FAVORABLE")
    emoji = _ADVISORY_EMOJI.get(worst, "⚪")

    lines = [
        f"# Tropi-Climate-Analytics — Seasonal Water Availability Forecast",
        f"**Issued:** {issue_date:%d %B %Y}  |  "
        f"**National Advisory:** {emoji} {worst}  |  "
        f"**Horizon:** 6 months",
        "",
        "## Watershed Forecasts",
        "",
        "| Watershed | Advisory | Avg Monthly Precip (mm) | Deficit Months | Confidence (%) | SPEI-3 Adj |",
        "|-----------|----------|------------------------|----------------|----------------|------------|",
    ]
    for f in outlook.get("forecasts", []):
        avg_precip = (sum(f["monthly_precip_mm"]) / len(f["monthly_precip_mm"])
                      if f["monthly_precip_mm"] else 0.0)
        deficit_n  = sum(f["water_deficit_months"])
        adv_emoji  = _ADVISORY_EMOJI.get(f["agricultural_advisory"], "⚪")
        lines.append(
            f"| {f['watershed_id'].replace('_',' ').title()} "
            f"| {adv_emoji} {f['agricultural_advisory']} "
            f"| {avg_precip:.0f} "
            f"| {deficit_n}/{f['horizon_months']} "
            f"| {f['confidence_band_pct']:.0f}% "
            f"| {'Yes' if f['spei_adjusted'] else 'No'} |"
        )

    lines += [
        "",
        "## National Rollup",
        f"- **Deficit/Critical watersheds:** "
        f"{outlook.get('deficit_watersheds', 0)} / {len(_WATERSHEDS)}",
        f"- **Average deficit months (6-month horizon):** "
        f"{outlook.get('total_deficit_months_avg', 0):.1f}",
        "",
        "## Agricultural Advisory",
        "| Level | Meaning |",
        "|-------|---------|",
        "| 🟢 FAVORABLE | Normal to above-normal water availability — proceed with planned crops |",
        "| 🟡 CAUTION   | Slight deficit possible — consider drought-tolerant varieties |",
        "| 🟠 DEFICIT   | Below-normal runoff likely — activate water-saving protocols |",
        "| 🔴 CRITICAL  | Severe water stress expected — coordinate with BBWS for irrigation priority |",
        "",
        "> *Generated by HYDROLOGIS hydrologis_seasonal_water_forecast · "
        "Tropi-Climate-Analytics*",
    ]

    summary_path.write_text("
".join(lines))
    logger.info("Seasonal summary written | path=%s", summary_path)
    ti.xcom_push(key="summary_path", value=str(summary_path))
    return {"summary_path": str(summary_path)}


def emit_forecast_metrics(**context) -> None:
    ti        = context["ti"]
    outlook   = ti.xcom_pull(key="national_outlook",
                              task_ids="aggregate_national_water_outlook") or {}
    try:
        from src.hydrology.metrics import SEASONAL_FORECAST_SKILL
        # Emit current confidence as proxy for skill (real skill updated in verify step)
        for f in outlook.get("forecasts", []):
            SEASONAL_FORECAST_SKILL.labels(
                watershed_id=f["watershed_id"],
                horizon_months="6",
            ).set(f["confidence_band_pct"] / 100.0)
        logger.info("SEASONAL_FORECAST_SKILL emitted for %d watersheds",
                    len(outlook.get("forecasts", [])))
    except Exception as exc:
        logger.warning("Metrics emit non-fatal: %s", exc)


def verify_prior_month_skill(**context) -> dict:
    """
    Compare last month's 1-month-ahead forecast vs observed QPE.
    Writes Pearson r skill scores to workspace/output/seasonal_forecast/skill_scores.json.
    Updates SEASONAL_FORECAST_SKILL gauge with verified skill (horizon_months='1').
    """
    import calendar as cal
    today       = date.today()
    prior_month = (today.replace(day=1) - timedelta(days=1))
    prior_ym    = prior_month.strftime("%Y%m")
    results     = {}

    for ws_id in _WATERSHEDS:
        fc_path  = _SF_DIR / f"forecast_{ws_id}_{prior_ym}.json"
        # Look for observed QPE aggregate (written by qpe_pipeline or batch summary)
        obs_path = _QPE_DIR / f"monthly_{ws_id}_{prior_ym}.json"

        if not fc_path.exists():
            logger.info("No prior forecast for %s %s — skip verification", ws_id, prior_ym)
            continue

        fc_data  = json.loads(fc_path.read_text())
        forecast_precip = fc_data.get("monthly_precip_mm", [None])[0]

        if obs_path.exists():
            obs_data  = json.loads(obs_path.read_text())
            obs_precip = obs_data.get("total_precip_mm")
        else:
            obs_precip = None  # verification deferred until QPE monthly file available

        skill_r = None
        if forecast_precip is not None and obs_precip is not None:
            # Single-point r is undefined; store ratio as skill proxy
            ratio   = obs_precip / max(forecast_precip, 1.0)
            skill_r = max(0.0, 1.0 - abs(ratio - 1.0))  # 1.0 = perfect
        results[ws_id] = {"forecast_mm": forecast_precip,
                           "observed_mm": obs_precip,
                           "skill_r":    skill_r,
                           "month":      prior_ym}

    scores_path = _SF_DIR / "skill_scores.json"
    existing    = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    existing.update(results)
    _SF_DIR.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(existing, indent=2))
    logger.info("Skill scores updated | %d watersheds | path=%s", len(results), scores_path)

    # Emit verified skill gauge
    try:
        from src.hydrology.metrics import SEASONAL_FORECAST_SKILL
        for ws_id, r in results.items():
            if r["skill_r"] is not None:
                SEASONAL_FORECAST_SKILL.labels(
                    watershed_id=ws_id, horizon_months="1"
                ).set(r["skill_r"])
    except Exception as exc:
        logger.debug("Skill gauge emit: %s", exc)

    return results


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="hydrologis_seasonal_water_forecast",
    description="Monthly seasonal water availability forecast: 5 watersheds, 6-month horizon, "
                "VISUALIA summary, prior-month skill verification.",
    schedule_interval="0 5 1 * *",
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    tags=["hydrologis", "seasonal", "forecast", "sprint9"],
    doc_md=__doc__,
) as dag:

    t_drought = PythonOperator(
        task_id="assess_drought_context",
        python_callable=assess_drought_context,
        sla=timedelta(minutes=3),
    )

    with TaskGroup("forecast_all_watersheds",
                   tooltip="Parallel seasonal forecast per watershed") as tg_fc:
        for _ws in _WATERSHEDS:
            PythonOperator(
                task_id=f"forecast_{_ws}",
                python_callable=_make_forecast_callable(_ws),
                sla=timedelta(minutes=12),
            )

    t_aggregate = PythonOperator(
        task_id="aggregate_national_water_outlook",
        python_callable=aggregate_national_water_outlook,
        sla=timedelta(minutes=18),
    )

    t_summary = PythonOperator(
        task_id="write_seasonal_summary",
        python_callable=write_seasonal_summary,
        sla=timedelta(minutes=22),
    )

    t_metrics = PythonOperator(
        task_id="emit_forecast_metrics",
        python_callable=emit_forecast_metrics,
        sla=timedelta(minutes=25),
    )

    t_verify = PythonOperator(
        task_id="verify_prior_month_skill",
        python_callable=verify_prior_month_skill,
        sla=timedelta(minutes=30),
    )

    t_drought >> tg_fc >> t_aggregate >> t_summary >> t_metrics >> t_verify
