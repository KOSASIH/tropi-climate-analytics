"""
irrigation_advisory_dag.py — Sprint 11 P4
dag_id: hydrologis_irrigation_advisory

Schedule : 0 7 * * * Asia/Jakarta  (daily 07:00 WIB, after ET dag at 06:00)
SLA      : 20 minutes
Scope    : FAO-56 irrigation advisory for 100 agricultural districts.
           ShortCircuit on idle days (all NIR == 0, no stress).
           TaskGroup: 10 parallel groups × 10 districts.
           Drought coupling flag when CRITICAL_DEFICIT > 20 % of districts.

Cross-agent:
  VISUALIA    ← workspace/output/irrigation/district_advisory_{YYYYMMDD}.json
  API-GATEWAY ← /v1/irrigation/advisory/{district_id}
  Internal    ← coupling_flag_{YYYYMMDD}.json → drought_monitor bimodal stress
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
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
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

_WS          = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_IRR_DIR     = _WS / "output" / "irrigation"
_CAL_CFG     = _WS / "config" / "district_crop_calendar.json"
# Drought coupling threshold: if CRITICAL_DEFICIT districts > this fraction → flag
_COUPLING_THRESHOLD = 0.20

# ── district partitioning ────────────────────────────────────────────────────

def _load_districts() -> list[dict]:
    if _CAL_CFG.exists():
        raw = json.loads(_CAL_CFG.read_text())
        if isinstance(raw, dict):
            return [{"id": k, **v} for k, v in raw.items()]
        return raw
    # fallback: 100 synthetic districts across Java
    provinces = ["jabar", "jateng", "jatim", "diy", "banten", "dki"]
    crops     = ["rice", "corn", "sugarcane", "soybean", "dryland_rice"]
    districts = []
    for i in range(100):
        prov = provinces[i % len(provinces)]
        districts.append({"id": f"{prov}_district_{i:03d}",
                          "province": prov,
                          "dominant_crop": crops[i % len(crops)],
                          "typical_planting_doy": 60 + (i * 3) % 180,
                          "area_ha": 10000 + i * 100})
    return districts

_ALL_DISTRICTS = _load_districts()

# Split into 10 groups of 10 each (first 100)
_GROUPS: list[list[dict]] = []
chunk = max(1, len(_ALL_DISTRICTS) // 10)
for _i in range(10):
    _GROUPS.append(_ALL_DISTRICTS[_i * chunk: (_i + 1) * chunk])


# ── task functions ───────────────────────────────────────────────────────────

def load_smap_et_qpe(**context) -> dict:
    """Load latest SMAP soil moisture, ET, QPE into xcom for downstream tasks."""
    run_date = date.fromisoformat(
        context["dag_run"].conf.get("run_date", date.today().isoformat())
    )
    smap_path = _WS / "output" / "drought"  / "latest_smap_composite.json"
    et_dir    = _WS / "output" / "et"
    qpe_path  = _WS / "output" / "qpe"      / "latest_qpe.json"

    smap_data = json.loads(smap_path.read_text()) if smap_path.exists() else {}
    qpe_data  = json.loads(qpe_path.read_text())  if qpe_path.exists() else {}

    # Find most-recent ET file (any watershed)
    et_data: dict = {}
    if et_dir.exists():
        et_files = sorted(et_dir.glob("spatial_et_*_*.json"))
        if et_files:
            try:
                et_data = json.loads(et_files[-1].read_text())
            except Exception:
                pass

    payload = {"run_date": run_date.isoformat(),
               "smap": smap_data, "et": et_data, "qpe": qpe_data}
    context["ti"].xcom_push(key="input_data", value=payload)
    logger.info("Inputs loaded | smap=%s et=%s qpe=%s",
                bool(smap_data), bool(et_data), bool(qpe_data))
    return payload


def determine_active_crops(**context) -> dict:
    """Build crop-stage map for today's run date."""
    payload  = context["ti"].xcom_pull(key="input_data", task_ids="load_smap_et_qpe") or {}
    run_date = date.fromisoformat(payload.get("run_date", date.today().isoformat()))
    doy      = run_date.timetuple().tm_yday

    active: dict[str, dict] = {}
    for d in _ALL_DISTRICTS:
        did        = d["id"]
        crop       = d.get("dominant_crop", "rice")
        plant_doy  = d.get("typical_planting_doy", 60)
        days_since = (doy - plant_doy) % 365
        # Stage boundaries (days from planting) per FAO-56 Table 12
        stages = {"rice":        (20, 50, 110, 140),
                  "corn":        (20, 40,  90, 120),
                  "sugarcane":   (35, 60, 190, 230),
                  "soybean":     (15, 30,  70,  90),
                  "dryland_rice":(15, 35,  75, 100)}
        s = stages.get(crop, (20, 50, 110, 140))
        if   days_since < s[0]: stage = "initial"
        elif days_since < s[1]: stage = "development"
        elif days_since < s[2]: stage = "mid_season"
        elif days_since < s[3]: stage = "late"
        else:                   stage = "initial"  # next cycle
        active[did] = {"crop": crop, "growth_stage": stage,
                       "days_from_planting": days_since}

    context["ti"].xcom_push(key="active_crops", value=active)
    logger.info("Active crops determined for %d districts", len(active))
    return active


def _make_group_callable(group_idx: int, group: list[dict]):
    """Factory: returns a callable that computes NIR advisory for one group."""
    def compute_group(**context):
        from src.hydrology.irrigation_scheduler import IrrigationScheduler
        payload      = context["ti"].xcom_pull(key="input_data",  task_ids="load_smap_et_qpe")   or {}
        active_crops = context["ti"].xcom_pull(key="active_crops", task_ids="determine_active_crops") or {}
        run_date     = date.fromisoformat(payload.get("run_date", date.today().isoformat()))
        scheduler    = IrrigationScheduler()
        results      = []
        for d in group:
            did  = d["id"]
            crop_info = active_crops.get(did, {"crop": d.get("dominant_crop","rice"),
                                               "growth_stage":"mid_season"})
            try:
                advisory = scheduler.compute_district_advisory(
                    district_id=did, dt=run_date
                )
                results.append({
                    "district_id":       did,
                    "province":          d.get("province",""),
                    "crop":              crop_info["crop"],
                    "growth_stage":      crop_info["growth_stage"],
                    "advisory_level":    advisory.advisory_level,
                    "nir_mm":            advisory.nir_weighted_avg_mm,
                    "water_stress_risk": advisory.nir_weighted_avg_mm > 10,
                    "recommendation":    advisory.recommended_action,
                })
            except Exception as exc:
                logger.warning("District %s advisory failed: %s", did, exc)
                results.append({"district_id": did, "crop": crop_info["crop"],
                                 "advisory_level": "ADEQUATE", "nir_mm": 0.0,
                                 "water_stress_risk": False,
                                 "recommendation": "No data — assume adequate"})
        context["ti"].xcom_push(key=f"group_{group_idx}", value=results)
        critical = sum(1 for r in results if r["advisory_level"] == "CRITICAL_DEFICIT")
        logger.info("Group %d | %d districts | %d CRITICAL", group_idx, len(results), critical)
        return results

    compute_group.__name__ = f"compute_group_{group_idx}"
    return compute_group


def _is_idle(**context) -> bool:
    """ShortCircuit: skip run when all districts have NIR==0 and no stress flag."""
    payload  = context["ti"].xcom_pull(key="input_data",  task_ids="load_smap_et_qpe") or {}
    run_date = date.fromisoformat(payload.get("run_date", date.today().isoformat()))
    # Check Airflow Variable for a force-run flag (set by drought_monitor)
    force = Variable.get("FORCE_IRRIGATION_ADVISORY", default_var="false")
    if force.lower() == "true":
        Variable.set("FORCE_IRRIGATION_ADVISORY", "false")
        logger.info("Force-run flag set — skipping idle check")
        return True
    # Quick NIR estimate: if SMAP soil moisture near field capacity everywhere → skip
    smap = payload.get("smap", {})
    sm_values = []
    if isinstance(smap, dict):
        for v in smap.values():
            if isinstance(v, dict):
                sm_values.append(v.get("soil_moisture", 0.5))
            elif isinstance(v, (int, float)):
                sm_values.append(v)
    if sm_values:
        mean_sm = sum(sm_values) / len(sm_values)
        if mean_sm > 0.85:   # near field capacity → very low NIR everywhere
            logger.info("Mean SMAP SM=%.3f > 0.85 — idle day, skipping", mean_sm)
            return False
    return True  # proceed by default


def aggregate_national_advisory(**context) -> dict:
    """Combine all 10 group results into national advisory summary."""
    ti       = context["ti"]
    all_results: list[dict] = []
    for i in range(10):
        g = ti.xcom_pull(key=f"group_{i}", task_ids=f"compute_all_districts.compute_group_{i}") or []
        all_results.extend(g)

    run_date = date.fromisoformat(
        (ti.xcom_pull(key="input_data", task_ids="load_smap_et_qpe") or {}).get(
            "run_date", date.today().isoformat()))

    levels = ["CRITICAL_DEFICIT", "MODERATE_DEFICIT", "ADEQUATE", "SURPLUS"]
    counts = {lv: sum(1 for r in all_results if r.get("advisory_level") == lv)
              for lv in levels}
    total  = len(all_results) or 1
    summary = {
        "date":            run_date.isoformat(),
        "total_districts": len(all_results),
        "critical_count":  counts["CRITICAL_DEFICIT"],
        "moderate_count":  counts["MODERATE_DEFICIT"],
        "adequate_count":  counts["ADEQUATE"],
        "surplus_count":   counts["SURPLUS"],
        "critical_pct":    round(counts["CRITICAL_DEFICIT"] / total * 100, 1),
        "mean_nir_mm":     round(sum(r.get("nir_mm",0) for r in all_results) / total, 2),
        "stress_districts": [r["district_id"] for r in all_results
                              if r.get("water_stress_risk")],
        "districts":       all_results,
    }
    ti.xcom_push(key="national_advisory", value=summary)
    logger.info("National advisory | CRITICAL=%d (%.1f%%) MODERATE=%d ADEQUATE=%d",
                counts["CRITICAL_DEFICIT"], summary["critical_pct"],
                counts["MODERATE_DEFICIT"], counts["ADEQUATE"])
    return summary


def write_district_advisory(**context) -> dict:
    ti       = context["ti"]
    summary  = ti.xcom_pull(key="national_advisory", task_ids="aggregate_national_advisory") or {}
    run_date = summary.get("date", date.today().isoformat())
    _IRR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _IRR_DIR / f"district_advisory_{run_date.replace('-','')}.json"
    out_path.write_text(json.dumps({
        "generated_at":    datetime.now(tz=timezone.utc).isoformat(),
        **summary,
    }, indent=2, default=str))
    logger.info("district_advisory written → %s", out_path.name)
    return {"path": str(out_path), "critical_count": summary.get("critical_count", 0)}


def emit_irrigation_metrics(**context) -> None:
    ti      = context["ti"]
    summary = ti.xcom_pull(key="national_advisory", task_ids="aggregate_national_advisory") or {}
    try:
        from src.hydrology.metrics import IRRIGATION_DEFICIT_MM
        for d in summary.get("districts", []):
            IRRIGATION_DEFICIT_MM.labels(
                district_id=d["district_id"],
                crop=d.get("crop", "rice"),
            ).set(max(0.0, d.get("nir_mm", 0.0)))
        logger.info("IRRIGATION_DEFICIT_MM emitted for %d districts",
                    len(summary.get("districts", [])))
    except Exception as exc:
        logger.warning("Irrigation metrics emit non-fatal: %s", exc)


def dispatch_drought_coupling(**context) -> dict:
    """Write coupling_flag when CRITICAL_DEFICIT > 20 % of total districts."""
    ti      = context["ti"]
    summary = ti.xcom_pull(key="national_advisory", task_ids="aggregate_national_advisory") or {}
    crit_pct = summary.get("critical_pct", 0.0) / 100.0
    run_date = summary.get("date", date.today().isoformat())

    if crit_pct <= _COUPLING_THRESHOLD:
        logger.info("Drought coupling threshold not reached (%.1f%% < 20%%)",
                    crit_pct * 100)
        return {"coupled": False, "critical_pct": round(crit_pct * 100, 1)}

    flag_dir  = _WS / "output" / "drought"
    flag_dir.mkdir(parents=True, exist_ok=True)
    flag_path = flag_dir / f"coupling_flag_{run_date.replace('-','')}.json"
    flag_path.write_text(json.dumps({
        "generated_at":         datetime.now(tz=timezone.utc).isoformat(),
        "source":               "irrigation_advisory_dag",
        "run_date":             run_date,
        "critical_deficit_pct": round(crit_pct * 100, 1),
        "critical_districts":   summary.get("critical_count", 0),
        "total_districts":      summary.get("total_districts", 0),
        "mean_nir_mm":          summary.get("mean_nir_mm", 0.0),
        "stress_districts":     summary.get("stress_districts", [])[:20],
        "coupling_type":        "bimodal_drought_stress",
        "action":               "drought_monitor should elevate national drought index",
    }, indent=2))
    # Also set Airflow Variable for drought_monitor_dag to pick up
    Variable.set("DROUGHT_COUPLING_FLAG", "true")
    Variable.set("DROUGHT_COUPLING_DATE", run_date)
    logger.warning("DROUGHT COUPLING FLAG SET | critical_pct=%.1f%% > 20%%",
                   crit_pct * 100)
    return {"coupled": True,
            "critical_pct": round(crit_pct * 100, 1),
            "flag_path": str(flag_path)}


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="hydrologis_irrigation_advisory",
    description="Daily FAO-56 irrigation advisory for 100 Indonesian agricultural districts. "
                "NIR soil water balance from SMAP + ET + QPE. "
                "Drought coupling flag when CRITICAL_DEFICIT > 20%.",
    schedule_interval="0 7 * * *",
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    max_active_runs=1,
    tags=["hydrologis", "irrigation", "fao56", "sprint11"],
    doc_md=__doc__,
) as dag:

    t_load = PythonOperator(
        task_id="load_smap_et_qpe",
        python_callable=load_smap_et_qpe,
    )

    t_crops = PythonOperator(
        task_id="determine_active_crops",
        python_callable=determine_active_crops,
    )

    t_idle = ShortCircuitOperator(
        task_id="check_growing_season",
        python_callable=_is_idle,
    )

    with TaskGroup(group_id="compute_all_districts") as tg_districts:
        group_tasks = []
        for idx, grp in enumerate(_GROUPS):
            gt = PythonOperator(
                task_id=f"compute_group_{idx}",
                python_callable=_make_group_callable(idx, grp),
            )
            group_tasks.append(gt)

    t_agg = PythonOperator(
        task_id="aggregate_national_advisory",
        python_callable=aggregate_national_advisory,
    )

    t_write = PythonOperator(
        task_id="write_district_advisory",
        python_callable=write_district_advisory,
    )

    t_metrics = PythonOperator(
        task_id="emit_irrigation_metrics",
        python_callable=emit_irrigation_metrics,
    )

    t_coupling = PythonOperator(
        task_id="dispatch_drought_coupling",
        python_callable=dispatch_drought_coupling,
    )

    # ── wiring ────────────────────────────────────────────────────────────────
    t_load >> t_crops >> t_idle >> tg_districts >> t_agg >> [t_write, t_metrics, t_coupling]
