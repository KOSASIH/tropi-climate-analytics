"""
evapotranspiration_dag.py — Sprint 10 M2
dag_id: hydrologis_evapotranspiration

Schedule: 0 6 * * * Asia/Jakarta  (daily 06:00 WIB — after QPE overnight cycle)
SLA: 15 minutes

Computes daily reference and spatial evapotranspiration for 5 strategic watersheds
using FAO-56 Penman-Monteith (ETPenmanMonteith). MODIS MOD16A2 8-day ET used for
cross-validation only (non-blocking ShortCircuit). ET outputs feed watershed_water_balance
and seasonal_water_forecast for water balance closure.

Watersheds: citarum, brantas, solo, musi, kapuas

Tasks:
    load_met_observations       → BMKG 24-h station obs from FeatureStoreClient
    load_modis_et               → ShortCircuit if MODIS tile stale > 10 days (non-blocking)
    compute_all_watersheds      (TaskGroup, 5 parallel)
      ├─ compute_citarum
      ├─ compute_brantas
      ├─ compute_solo
      ├─ compute_musi
      └─ compute_kapuas
    aggregate_national_et       → national mean/max/min ETo; high-stress flag (ETo > 5mm/day)
    write_et_outputs            → spatial_et_{watershed_id}_{YYYYMM}.json
    emit_et_metrics             → ET_DAILY_MM{watershed_id, method} Gauge

Cross-agent handoff:
    Internal: workspace/output/et/ → watershed_water_balance.py + seasonal_water_forecast.py
    VISUALIA: workspace/output/et/spatial_et_{watershed_id}_{YYYYMM}.json (daily)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "hydrologis",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

_WS             = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_ET_DIR         = _WS / "output" / "et"
_MODIS_DIR      = _WS / "data" / "modis"
_MODIS_STALE_S  = 10 * 24 * 3600  # 10 days — MODIS MOD16A2 is 8-day composite
_WATERSHEDS     = ["citarum", "brantas", "solo", "musi", "kapuas"]
_HIGH_STRESS_MM = 5.0   # ETo > 5 mm/day → high evaporative demand


# ── staleness check ───────────────────────────────────────────────────────────

def load_met_observations(**context) -> dict:
    """Load last-24h BMKG station observations from FeatureStoreClient."""
    obs_map: dict = {}
    run_date = date.fromisoformat(
        context["dag_run"].conf.get("run_date", date.today().isoformat())
    )
    try:
        from src.data.feature_store import FeatureStoreClient
        fs = FeatureStoreClient()
        for ws_id in _WATERSHEDS:
            obs = fs.get_latest_obs(entity_type="station_id",
                                    region=ws_id,
                                    lookback_hours=24,
                                    as_of=run_date)
            obs_map[ws_id] = obs
        logger.info("BMKG obs loaded for %d watersheds", len(obs_map))
    except Exception as exc:
        logger.warning("FeatureStoreClient unavailable — synthetic fallback: %s", exc)
        # ETPenmanMonteith._synthetic_bmkg() handles missing obs internally
        obs_map = {ws_id: None for ws_id in _WATERSHEDS}

    context["ti"].xcom_push(key="met_obs", value=obs_map)
    return obs_map


def load_modis_et(**context) -> bool:
    """
    ShortCircuitOperator: returns True even when stale.
    MODIS is cross-validation only — never blocks the pipeline.
    Just check and log; downstream tasks handle missing MODIS gracefully.
    """
    run_date = date.fromisoformat(
        context["dag_run"].conf.get("run_date", date.today().isoformat())
    )
    # Look for any MOD16A2 tile within the staleness window
    tiles = sorted(_MODIS_DIR.glob("MOD16A2_*.tif")) if _MODIS_DIR.exists() else []
    if tiles:
        age_s = (datetime.now().timestamp() - tiles[-1].stat().st_mtime)
        if age_s <= _MODIS_STALE_S:
            logger.info("MODIS MOD16A2 tile fresh (age=%.0fh)", age_s / 3600)
            context["ti"].xcom_push(key="modis_available", value=True)
        else:
            logger.info("MODIS MOD16A2 tile stale (age=%.1fd) — cross-val skipped",
                        age_s / 86400)
            context["ti"].xcom_push(key="modis_available", value=False)
    else:
        logger.info("No MODIS MOD16A2 tile found — cross-val skipped")
        context["ti"].xcom_push(key="modis_available", value=False)
    return True  # always pass — MODIS is non-blocking


# ── per-watershed ET factory ──────────────────────────────────────────────────

def _make_et_callable(watershed_id: str):
    def compute_et(**context):
        from src.hydrology.evapotranspiration import ETPenmanMonteith
        ti = context["ti"]
        run_date = date.fromisoformat(
            context["dag_run"].conf.get("run_date", date.today().isoformat())
        )
        modis_ok = ti.xcom_pull(key="modis_available", task_ids="load_modis_et") or False

        et = ETPenmanMonteith()
        result = et.compute_spatial(watershed_id=watershed_id, date=run_date)

        payload = {
            "watershed_id":       result.watershed_id,
            "date":               result.date.isoformat(),
            "mean_et0_mm":        result.mean_et0_mm,
            "modis_et_mm":        result.modis_et_mm if modis_ok else None,
            "et_deficit_mm":      result.et_deficit_mm if modis_ok else None,
            "n_stations":         result.n_stations,
            "coverage_pct":       result.coverage_pct,
            "high_stress":        result.mean_et0_mm > _HIGH_STRESS_MM,
        }
        ti.xcom_push(key=f"et_{watershed_id}", value=payload)
        logger.info("ET computed | %s date=%s et0=%.2fmm stress=%s",
                    watershed_id, run_date, result.mean_et0_mm,
                    payload["high_stress"])
        return payload
    compute_et.__name__ = f"compute_{watershed_id}"
    return compute_et


# ── downstream tasks ──────────────────────────────────────────────────────────

def aggregate_national_et(**context) -> dict:
    ti = context["ti"]
    results = []
    for ws_id in _WATERSHEDS:
        r = ti.xcom_pull(
            key=f"et_{ws_id}",
            task_ids=f"compute_all_watersheds.compute_{ws_id}",
        )
        if r:
            results.append(r)

    if not results:
        logger.warning("No ET results to aggregate")
        return {}

    et0_vals     = [r["mean_et0_mm"] for r in results]
    stress_count = sum(1 for r in results if r["high_stress"])
    outlook = {
        "date":                  results[0]["date"],
        "national_mean_et0_mm":  sum(et0_vals) / len(et0_vals),
        "national_max_et0_mm":   max(et0_vals),
        "national_min_et0_mm":   min(et0_vals),
        "high_stress_watersheds": stress_count,
        "watersheds":            results,
    }
    if stress_count > 0:
        ws_names = [r["watershed_id"] for r in results if r["high_stress"]]
        logger.warning("HIGH ET STRESS | %d/%d watersheds: %s",
                       stress_count, len(_WATERSHEDS), ws_names)
    ti.xcom_push(key="national_et", value=outlook)
    return outlook


def write_et_outputs(**context) -> dict:
    ti       = context["ti"]
    outlook  = ti.xcom_pull(key="national_et", task_ids="aggregate_national_et") or {}
    run_date = date.fromisoformat(
        context["dag_run"].conf.get("run_date", date.today().isoformat())
    )
    _ET_DIR.mkdir(parents=True, exist_ok=True)

    paths = []
    for ws_data in outlook.get("watersheds", []):
        ym   = run_date.strftime("%Y%m")
        path = _ET_DIR / f"spatial_et_{ws_data['watershed_id']}_{ym}.json"
        # Append to monthly file (or create)
        existing: list = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        # Replace today's entry if already present, else append
        existing = [e for e in existing if e.get("date") != ws_data["date"]]
        existing.append(ws_data)
        path.write_text(json.dumps(existing, indent=2))
        paths.append(str(path))

    logger.info("ET outputs written | %d files", len(paths))
    return {"files_written": len(paths), "date": run_date.isoformat()}


def emit_et_metrics(**context) -> None:
    ti      = context["ti"]
    outlook = ti.xcom_pull(key="national_et", task_ids="aggregate_national_et") or {}
    try:
        from src.hydrology.metrics import ET_DAILY_MM
        for ws in outlook.get("watersheds", []):
            ET_DAILY_MM.labels(
                watershed_id=ws["watershed_id"], method="penman_monteith"
            ).set(ws["mean_et0_mm"])
            if ws.get("modis_et_mm") is not None:
                ET_DAILY_MM.labels(
                    watershed_id=ws["watershed_id"], method="modis"
                ).set(ws["modis_et_mm"])
        logger.info("ET_DAILY_MM metrics emitted for %d watersheds",
                    len(outlook.get("watersheds", [])))
    except Exception as exc:
        logger.warning("ET metrics emit non-fatal: %s", exc)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="hydrologis_evapotranspiration",
    description="Daily FAO-56 Penman-Monteith ET for 5 watersheds with MODIS cross-val. "
                "Feeds watershed water balance and seasonal forecast.",
    schedule_interval="0 6 * * *",
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    tags=["hydrologis", "evapotranspiration", "water_balance", "sprint10"],
    doc_md=__doc__,
) as dag:

    t_met = PythonOperator(
        task_id="load_met_observations",
        python_callable=load_met_observations,
        sla=timedelta(minutes=2),
    )

    t_modis = ShortCircuitOperator(
        task_id="load_modis_et",
        python_callable=load_modis_et,
        sla=timedelta(minutes=3),
    )

    with TaskGroup("compute_all_watersheds",
                   tooltip="Parallel ET computation per watershed") as tg_et:
        for _ws in _WATERSHEDS:
            PythonOperator(
                task_id=f"compute_{_ws}",
                python_callable=_make_et_callable(_ws),
                sla=timedelta(minutes=8),
            )

    t_agg = PythonOperator(
        task_id="aggregate_national_et",
        python_callable=aggregate_national_et,
        sla=timedelta(minutes=10),
    )

    t_write = PythonOperator(
        task_id="write_et_outputs",
        python_callable=write_et_outputs,
        sla=timedelta(minutes=12),
    )

    t_metrics = PythonOperator(
        task_id="emit_et_metrics",
        python_callable=emit_et_metrics,
        sla=timedelta(minutes=15),
    )

    [t_met, t_modis] >> tg_et >> t_agg >> t_write >> t_metrics
