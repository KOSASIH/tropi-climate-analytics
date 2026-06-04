"""
reservoir_operations_dag.py — Sprint 10 M4
dag_id: hydrologis_reservoir_operations

Schedule: 0 * * * * Asia/Jakarta  (hourly — reservoir routing needs hourly update)
SLA: 8 minutes

Rule-curve-based routing for 5 major Indonesian reservoirs using ReservoirOperationsModel.
Integrates latest streamflow forecast (HYDROLOGIS streamflow DAG) + QPE accumulation
as inflow proxies. Dispatches BPBD gate alerts for EMERGENCY status with 1-hour dedup window.
Writes active_operations.json rolling sidecar for GEOSPATIAL + VISUALIA.

Reservoirs: jatiluhur, saguling, cirata, gajah_mungkur, upper_cisokan

Tasks:
    load_reservoir_levels       → latest per-reservoir JSON (or config defaults)
    load_inflow_forecast        → streamflow latest_forecast.json + QPE for citarum cascade
    compute_all_reservoirs      (TaskGroup, 5 parallel)
      ├─ compute_jatiluhur
      ├─ compute_saguling
      ├─ compute_cirata
      ├─ compute_gajah_mungkur
      └─ compute_upper_cisokan
    write_active_operations     → active_operations.json sidecar
    emit_reservoir_metrics      → RESERVOIR_STORAGE_FRACTION{reservoir_id} Gauge
    dispatch_gate_alerts        → BPBD webhook for EMERGENCY / MWL breach, 1hr dedup

Cross-agent handoff:
    GEOSPATIAL + VISUALIA: workspace/output/reservoir/active_operations.json (hourly)
    flood_early_warning_dag: active_operations.json (downstream inflow modifier)
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
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "hydrologis",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
}

_WS           = Path(os.environ.get("HYDRO_WORKSPACE", "/opt/airflow/workspace"))
_RES_DIR      = _WS / "output" / "reservoir"
_SF_PATH      = _WS / "output" / "streamflow" / "latest_forecast.json"
_QPE_PATH     = _WS / "output" / "qpe" / "latest_qpe.json"
_RESERVOIRS   = ["jatiluhur", "saguling", "cirata", "gajah_mungkur", "upper_cisokan"]
_ALERT_STATUSES = {"ALERT", "EMERGENCY"}
_DEDUP_TTL_S  = 3600  # 1-hour gate alert dedup


# ── data loading ──────────────────────────────────────────────────────────────

def load_reservoir_levels(**context) -> dict:
    """
    Load latest stored state per reservoir. Falls back to config-derived initial
    state (50% active capacity) if no prior run JSON found.
    """
    levels: dict = {}
    for res_id in _RESERVOIRS:
        # Most recent hourly JSON for this reservoir
        pattern = sorted(_RES_DIR.glob(f"reservoir_{res_id}_*.json")) if _RES_DIR.exists() else []
        if pattern:
            data = json.loads(pattern[-1].read_text())
            levels[res_id] = {
                "storage_m3":       data.get("storage_m3"),
                "storage_fraction": data.get("storage_fraction"),
                "inflow_cms":       data.get("inflow_cms", 0.0),
                "outflow_cms":      data.get("outflow_cms", 0.0),
            }
            logger.info("Loaded state | %s storage_frac=%.3f", res_id,
                        data.get("storage_fraction", 0))
        else:
            levels[res_id] = None  # will use default init in compute step
            logger.info("No prior state for %s — using defaults", res_id)
    context["ti"].xcom_push(key="reservoir_levels", value=levels)
    return levels


def load_inflow_forecast(**context) -> dict:
    """
    Load streamflow 6hr forecast + QPE accumulation as inflow estimates per reservoir.
    citarum cascade (saguling → cirata → jatiluhur) gets QPE supplement.
    """
    sf_data: dict = {}
    qpe_data: dict = {}

    if _SF_PATH.exists():
        try:
            sf_data = json.loads(_SF_PATH.read_text())
        except Exception as exc:
            logger.warning("Failed to read streamflow forecast: %s", exc)

    if _QPE_PATH.exists():
        try:
            qpe_data = json.loads(_QPE_PATH.read_text())
        except Exception as exc:
            logger.warning("Failed to read QPE: %s", exc)

    # Build inflow map: river → cms (from streamflow, fallback climatological)
    _RIVER_FLOW_MAP = {
        "citarum":      sf_data.get("citarum_forecast_cms_6hr", 250.0),
        "bengawan_solo": sf_data.get("bengawan_solo_forecast_cms_6hr", 150.0),
        "cisokan":      sf_data.get("cisokan_forecast_cms_6hr", 60.0),
    }
    # QPE supplement for Citarum cascade: add runoff proxy from basin QPE
    citarum_qpe_mm = qpe_data.get("citarum_basin_qpe_1hr_mm", 0.0)
    # Simple runoff coefficient: area_citarum_km2 × qpe × 0.6 / 3600 = cms
    qpe_supplement_cms = 4_500 * citarum_qpe_mm * 0.6 / 3600.0
    _RIVER_FLOW_MAP["citarum"] += qpe_supplement_cms

    inflow = {
        "jatiluhur":     _RIVER_FLOW_MAP["citarum"] * 0.85,   # downstream of cascade
        "saguling":      _RIVER_FLOW_MAP["citarum"] * 1.00,   # most upstream citarum
        "cirata":        _RIVER_FLOW_MAP["citarum"] * 0.95,   # mid-cascade
        "gajah_mungkur": _RIVER_FLOW_MAP["bengawan_solo"],
        "upper_cisokan": _RIVER_FLOW_MAP["cisokan"],
    }
    logger.info("Inflow map: %s", {k: f"{v:.1f}" for k, v in inflow.items()})
    context["ti"].xcom_push(key="inflow_forecast", value=inflow)
    return inflow


# ── per-reservoir compute factory ────────────────────────────────────────────

def _make_reservoir_callable(reservoir_id: str):
    def compute_reservoir(**context):
        from src.hydrology.reservoir_operations import ReservoirOperationsModel
        ti      = context["ti"]
        levels  = ti.xcom_pull(key="reservoir_levels",  task_ids="load_reservoir_levels") or {}
        inflows = ti.xcom_pull(key="inflow_forecast",   task_ids="load_inflow_forecast") or {}

        model        = ReservoirOperationsModel()
        inflow_cms   = inflows.get(reservoir_id, 50.0)
        prior_state  = levels.get(reservoir_id)

        # Derive current storage from prior state or model default
        if prior_state and prior_state.get("storage_m3") is not None:
            current_storage_m3 = prior_state["storage_m3"]
            prior_outflow_cms  = prior_state.get("outflow_cms", inflow_cms)
        else:
            # Default: 55% active capacity (between TOL and FSL)
            cfg = model._get_cfg(reservoir_id)
            current_storage_m3 = cfg["dead_storage_m3"] + 0.55 * cfg["active_capacity_m3"]
            prior_outflow_cms  = inflow_cms

        # Hourly routing (dt=1hr)
        storage_update = model.update_storage(
            reservoir_id=reservoir_id,
            inflow_cms=inflow_cms,
            outflow_cms=prior_outflow_cms,
            dt_hours=1.0,
        )
        # Gate decision based on current storage after update
        gate_decision = model.compute_flood_gate_release(
            reservoir_id=reservoir_id,
            current_storage_m3=storage_update.storage_m3,
            inflow_forecast_cms=[inflow_cms] * 6,  # flat 6hr forecast
        )
        # Downstream impact if alert or emergency
        impact = None
        if gate_decision.gate_status in ("ALERT", "EMERGENCY"):
            impact = model.get_downstream_impact(
                reservoir_id=reservoir_id,
                release_cms=gate_decision.recommended_release_cms,
            )

        payload = {
            "reservoir_id":         reservoir_id,
            "timestamp":            datetime.now(tz=timezone.utc).isoformat(),
            "storage_m3":           storage_update.storage_m3,
            "storage_fraction":     storage_update.storage_fraction,
            "level_masl":           storage_update.level_masl,
            "inflow_cms":           storage_update.inflow_cms,
            "outflow_cms":          storage_update.outflow_cms,
            "gate_status":          storage_update.gate_status,
            "water_supply_risk":    storage_update.water_supply_risk,
            "recommended_release_cms": gate_decision.recommended_release_cms,
            "dispatch_alert":       gate_decision.dispatch_alert,
            "alert_level":          gate_decision.alert_level,
            "gate_rationale":       gate_decision.rationale,
            "downstream_impact":    {
                "peak_cms_24hr":   impact.peak_cms_24hr,
                "travel_time_hr":  impact.travel_time_hr,
                "affected_cities": impact.affected_cities,
            } if impact else None,
        }

        # Write hourly file
        _RES_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H")
        out_path = _RES_DIR / f"reservoir_{reservoir_id}_{ts_str}.json"
        out_path.write_text(json.dumps(payload, indent=2))

        ti.xcom_push(key=f"reservoir_{reservoir_id}", value=payload)
        logger.info("Reservoir | %s frac=%.3f status=%s",
                    reservoir_id, storage_update.storage_fraction, storage_update.gate_status)
        return payload
    compute_reservoir.__name__ = f"compute_{reservoir_id}"
    return compute_reservoir


# ── downstream tasks ──────────────────────────────────────────────────────────

def write_active_operations(**context) -> dict:
    ti = context["ti"]
    ops = []
    for res_id in _RESERVOIRS:
        r = ti.xcom_pull(
            key=f"reservoir_{res_id}",
            task_ids=f"compute_all_reservoirs.compute_{res_id}",
        )
        if r:
            ops.append(r)

    highest_status = "NORMAL"
    _rank = {"NORMAL": 0, "ALERT": 1, "EMERGENCY": 2}
    for op in ops:
        if _rank.get(op.get("gate_status", "NORMAL"), 0) > _rank.get(highest_status, 0):
            highest_status = op["gate_status"]

    sidecar = {
        "generated_at":     datetime.now(tz=timezone.utc).isoformat(),
        "reservoir_count":  len(ops),
        "highest_status":   highest_status,
        "alert_count":      sum(1 for op in ops if op.get("gate_status") in _ALERT_STATUSES),
        "operations":       ops,
    }
    _RES_DIR.mkdir(parents=True, exist_ok=True)
    (_RES_DIR / "active_operations.json").write_text(json.dumps(sidecar, indent=2))
    logger.info("active_operations.json written | status=%s alerts=%d",
                highest_status, sidecar["alert_count"])
    ti.xcom_push(key="active_operations", value=sidecar)
    return {"highest_status": highest_status, "alert_count": sidecar["alert_count"]}


def emit_reservoir_metrics(**context) -> None:
    ti  = context["ti"]
    ops = ti.xcom_pull(key="active_operations",
                        task_ids="write_active_operations") or {}
    try:
        from src.hydrology.metrics import RESERVOIR_STORAGE_FRACTION
        for op in ops.get("operations", []):
            RESERVOIR_STORAGE_FRACTION.labels(
                reservoir_id=op["reservoir_id"]
            ).set(op["storage_fraction"])
        logger.info("RESERVOIR_STORAGE_FRACTION emitted for %d reservoirs",
                    len(ops.get("operations", [])))
    except Exception as exc:
        logger.warning("Reservoir metrics emit non-fatal: %s", exc)


def dispatch_gate_alerts(**context) -> dict:
    """
    Dispatch BPBD webhook alerts for ALERT/EMERGENCY gate status.
    Dedup: Airflow Variable RESERVOIR_ALERT_{reservoir_id} stores last dispatch
    timestamp; skip re-dispatch within _DEDUP_TTL_S (1 hour).
    """
    ti  = context["ti"]
    ops = ti.xcom_pull(key="active_operations",
                        task_ids="write_active_operations") or {}
    alert_ops    = [op for op in ops.get("operations", [])
                    if op.get("gate_status") in _ALERT_STATUSES]
    dispatched   = []
    now_ts       = datetime.now(tz=timezone.utc).timestamp()

    for op in alert_ops:
        res_id   = op["reservoir_id"]
        var_key  = f"RESERVOIR_ALERT_{res_id.upper()}"
        last_str = Variable.get(var_key, default_var=None)

        if last_str and (now_ts - float(last_str)) < _DEDUP_TTL_S:
            logger.info("Dedup suppressed | %s last=%.0fm ago",
                        res_id, (now_ts - float(last_str)) / 60)
            continue

        try:
            from src.hydrology.reservoir_operations import ReservoirOperationsModel
            model = ReservoirOperationsModel()
            model._dispatch_reservoir_alert(
                reservoir_id=res_id,
                gate_status=op["gate_status"],
                storage_fraction=op["storage_fraction"],
                recommended_release_cms=op["recommended_release_cms"],
                downstream_impact=op.get("downstream_impact"),
            )
            Variable.set(var_key, str(now_ts))
            dispatched.append(res_id)
            logger.warning("BPBD gate alert dispatched | %s status=%s frac=%.3f",
                           res_id, op["gate_status"], op["storage_fraction"])
        except Exception as exc:
            logger.error("Gate alert dispatch failed for %s: %s", res_id, exc)

    return {"dispatched": dispatched, "suppressed": len(alert_ops) - len(dispatched)}


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="hydrologis_reservoir_operations",
    description="Hourly rule-curve reservoir routing: 5 reservoirs, gate decisions, "
                "1hr-dedup BPBD alerts, active_operations.json sidecar.",
    schedule_interval="0 * * * *",
    start_date=days_ago(1),
    default_args=_DEFAULT_ARGS,
    catchup=False,
    max_active_runs=1,
    tags=["hydrologis", "reservoir", "operations", "sprint10"],
    doc_md=__doc__,
) as dag:

    t_levels = PythonOperator(
        task_id="load_reservoir_levels",
        python_callable=load_reservoir_levels,
        sla=timedelta(minutes=1),
    )

    t_inflow = PythonOperator(
        task_id="load_inflow_forecast",
        python_callable=load_inflow_forecast,
        sla=timedelta(minutes=2),
    )

    with TaskGroup("compute_all_reservoirs",
                   tooltip="Parallel reservoir routing") as tg_res:
        for _res in _RESERVOIRS:
            PythonOperator(
                task_id=f"compute_{_res}",
                python_callable=_make_reservoir_callable(_res),
                sla=timedelta(minutes=5),
            )

    t_write = PythonOperator(
        task_id="write_active_operations",
        python_callable=write_active_operations,
        sla=timedelta(minutes=6),
    )

    t_metrics = PythonOperator(
        task_id="emit_reservoir_metrics",
        python_callable=emit_reservoir_metrics,
        sla=timedelta(minutes=7),
    )

    t_alert = PythonOperator(
        task_id="dispatch_gate_alerts",
        python_callable=dispatch_gate_alerts,
        sla=timedelta(minutes=8),
    )

    [t_levels, t_inflow] >> tg_res >> t_write >> t_metrics >> t_alert
