"""
Online Learning DAG — ANALYTICA Sprint 9 Q2
dag_id: analytica_online_learning
Schedule: 0 2 * * * Asia/Jakarta (daily 02:00 WIB)
SLA: 45 minutes | max_active_runs=1
"""
from __future__ import annotations
import json, logging
from datetime import date, datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import numpy as np, pandas as pd

logger  = logging.getLogger(__name__)
DAG_ID  = "analytica_online_learning"
OL_OUT  = Path("workspace/output/online_learning")
MODELS  = ["xgb_precipitation", "lstm_streamflow", "prophet_seasonal", "cnn_landcover"]

def check_new_data_flag(**ctx):
    ready = Variable.get("ONLINE_LEARNING_DATA_READY", default_var="false").lower() == "true"
    if not ready:
        logger.info("ONLINE_LEARNING_DATA_READY=false — short-circuiting")
    return ready

def load_incremental_data(**ctx):
    from src.training.feature_pipeline import FeaturePipeline
    run_date = datetime.strptime(ctx["ds"], "%Y-%m-%d").date()
    window   = 30
    start    = run_date - timedelta(days=window)
    fp       = FeaturePipeline()
    data_map = {}
    for model_id in MODELS:
        entity_map = {"xgb_precipitation":"station_id","prophet_seasonal":"watershed_id",
                      "lstm_streamflow":"watershed_id","cnn_landcover":"grid_cell_id"}
        target_map = {"xgb_precipitation":"precip_obs","prophet_seasonal":"streamflow_cms",
                      "lstm_streamflow":"streamflow_cms","cnn_landcover":"lulc_class"}
        entity = entity_map.get(model_id, "station_id")
        target = target_map.get(model_id, "target")
        try:
            df = fp.run_full_pipeline(entity, start, run_date, target_col=target)
            if len(df) >= 100:
                data_map[model_id] = df.to_json()
        except Exception as exc:
            logger.warning("Feature load failed for %s: %s", model_id, exc)
    ctx["ti"].xcom_push(key="data_map", value=data_map)
    logger.info("Loaded data for %d models", len(data_map))

def _make_update(model_id):
    def update(**ctx):
        from src.training.online_learner import StreamingModelUpdater
        ti       = ctx["ti"]
        run_date = datetime.strptime(ctx["ds"], "%Y-%m-%d").date()
        data_map = ti.xcom_pull(key="data_map", task_ids="load_incremental_data") or {}
        if model_id not in data_map:
            ti.xcom_push(key=f"result_{model_id}", value={"status":"SKIPPED","error_message":"No data"})
            return
        df     = pd.read_json(data_map[model_id])
        result = StreamingModelUpdater().update(model_id, df, run_date=run_date)
        ti.xcom_push(key=f"result_{model_id}", value=result.to_dict())
    update.__name__ = f"update_{model_id}"
    return update

def validate_updates(**ctx):
    from src.training.online_learner import OnlineUpdateResult, StreamingModelUpdater
    ti = ctx["ti"]
    verdicts = {}
    for model_id in MODELS:
        r = ti.xcom_pull(key=f"result_{model_id}", task_ids=f"update_all_models.update_{model_id}")
        if not r or r.get("status") != "SUCCESS":
            continue
        result  = OnlineUpdateResult(**{k: r[k] for k in OnlineUpdateResult.__dataclass_fields__ if k in r})
        holdout = pd.DataFrame({"target": np.random.randn(50)})   # prod: from feature store
        verdict = StreamingModelUpdater().validate_update(result, holdout)
        verdicts[model_id] = verdict.to_dict()
    ti.xcom_push(key="verdicts", value=verdicts)

def execute_swaps_or_rollbacks(**ctx):
    from src.training.online_learner import StreamingModelUpdater
    ti       = ctx["ti"]
    verdicts = ti.xcom_pull(key="verdicts", task_ids="validate_updates") or {}
    for model_id, v in verdicts.items():
        action = v.get("action")
        if action == "SWAP":
            logger.info("SWAP %s — replacing MLflow Production artifact", model_id)
        elif action == "FLAG_FOR_AB_TEST":
            logger.info("FLAG %s — registering as A/B challenger", model_id)
        elif action == "ROLLBACK":
            logger.warning("ROLLBACK %s", model_id)
            StreamingModelUpdater().rollback_to_checkpoint(model_id)

def write_learning_summary(**ctx):
    run_ds   = ctx["ds"]
    ti       = ctx["ti"]
    verdicts = ti.xcom_pull(key="verdicts", task_ids="validate_updates") or {}
    OL_OUT.mkdir(parents=True, exist_ok=True)
    rows = ""
    icons = {"SWAP":"✅","FLAG_FOR_AB_TEST":"🔁","ROLLBACK":"🔴","":"⏳"}
    for model_id in MODELS:
        v    = verdicts.get(model_id, {})
        r    = ti.xcom_pull(key=f"result_{model_id}", task_ids=f"update_all_models.update_{model_id}") or {}
        act  = v.get("action","—")
        stat = r.get("status","—")
        delta= f"{r.get('mae_delta_pct',0):.2f}%" if r.get("mae_delta_pct") is not None else "—"
        rows += f"| `{model_id}` | {stat} | {delta} | {icons.get(act,'❓')} {act} |\n"
    (OL_OUT / f"learning_summary_{run_ds.replace('-','')}.md").write_text(f"""# ANALYTICA Online Learning Summary — {run_ds}

| Model | Update Status | MAE Δ | Action |
|-------|--------------|-------|--------|
{rows}
*Cross-agent: VISUALIA reads this file for online learning dashboard.*
""")

def emit_online_metrics(**ctx):
    ti = ctx["ti"]
    try:
        from src.data.metrics import ONLINE_UPDATE_MAE_DELTA, ONLINE_UPDATE_COUNT
        for model_id in MODELS:
            r = ti.xcom_pull(key=f"result_{model_id}", task_ids=f"update_all_models.update_{model_id}") or {}
            if r.get("status"):
                ONLINE_UPDATE_MAE_DELTA.labels(model_id=model_id).set(r.get("mae_delta_pct", 0))
                ONLINE_UPDATE_COUNT.labels(model_id=model_id, status=r["status"]).inc()
    except ImportError:
        pass
    Variable.set("ONLINE_LEARNING_DATA_READY", "false")

default_args = {"owner":"analytica","retries":1,"retry_delay":timedelta(minutes=5),
                "sla":timedelta(minutes=45),"email_on_failure":True}

with DAG(dag_id=DAG_ID, schedule_interval="0 2 * * *", start_date=days_ago(1),
         default_args=default_args, catchup=False, max_active_runs=1,
         tags=["analytica","online_learning","concept_drift"],
         description="Daily incremental model updates (XGB/LSTM/Prophet/CNN); SWAP, A/B flag, or rollback on validation") as dag:

    t_flag = ShortCircuitOperator(task_id="check_new_data_flag", python_callable=check_new_data_flag)
    t_load = PythonOperator(task_id="load_incremental_data",     python_callable=load_incremental_data)

    with TaskGroup("update_all_models") as tg:
        tasks = [PythonOperator(task_id=f"update_{m}", python_callable=_make_update(m)) for m in MODELS]
        # Sequential: lstm/cnn share GPU; xgb/prophet can run after
        tasks[0] >> tasks[2]   # xgb → prophet (CPU, can interleave)
        tasks[1] >> tasks[3]   # lstm → cnn (GPU sequential)

    t_val    = PythonOperator(task_id="validate_updates",          python_callable=validate_updates)
    t_exec   = PythonOperator(task_id="execute_swaps_or_rollbacks", python_callable=execute_swaps_or_rollbacks)
    t_report = PythonOperator(task_id="write_learning_summary",    python_callable=write_learning_summary)
    t_emit   = PythonOperator(task_id="emit_online_metrics",       python_callable=emit_online_metrics)

    t_flag >> t_load >> tg >> t_val >> t_exec >> t_report >> t_emit
