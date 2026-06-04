"""
HPO DAG — ANALYTICA Sprint 8 N4
dag_id: analytica_hyperparameter_optimization
Schedule: 0 2 10 * * Asia/Jakarta (monthly, 10th 02:00 WIB)
Also event-driven: runs when RETRAIN_* Variables are true
SLA: 90 minutes | max_active_runs=1
"""
from __future__ import annotations
import json, logging
from datetime import date, datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger  = logging.getLogger(__name__)
DAG_ID  = "analytica_hyperparameter_optimization"
HPO_DIR = Path("workspace/output/hpo")

RETRAIN_FLAGS = {
    "xgb_precipitation": "RETRAIN_XGB",
    "prophet_seasonal":  "RETRAIN_PROPHET",
    "lstm_streamflow":   "RETRAIN_LSTM",
    "cnn_landcover":     "RETRAIN_CNN",
}

def check_retrain_flags(**ctx):
    models = []
    for model_id, var_name in RETRAIN_FLAGS.items():
        if Variable.get(var_name, default_var="false").lower() == "true":
            models.append(model_id)
    if not models:
        models = list(RETRAIN_FLAGS.keys())   # monthly full run
    ctx["ti"].xcom_push(key="models_to_optimize", value=models)
    logger.info("HPO targets: %s", models)

def _make_optimize(model_id: str):
    def optimize(**ctx):
        from src.training.hyperparameter_optimizer import HPOOptimizer
        run_date = datetime.strptime(ctx["ds"], "%Y-%m-%d").date()
        models   = ctx["ti"].xcom_pull(key="models_to_optimize", task_ids="check_retrain_flags") or []
        if model_id not in models:
            logger.info("Skipping HPO for %s (not in target list)", model_id)
            return
        result = HPOOptimizer().optimize(model_id, n_trials=50, timeout_s=3600, run_date=run_date)
        ctx["ti"].xcom_push(key=f"hpo_{model_id}", value=result.to_dict())
    optimize.__name__ = f"optimize_{model_id}"
    return optimize

def register_improved_models(**ctx):
    ti = ctx["ti"]
    for model_id in RETRAIN_FLAGS:
        r = ti.xcom_pull(key=f"hpo_{model_id}", task_ids=f"optimize_all_models.optimize_{model_id}")
        if r and r.get("promoted"):
            logger.info("HPO improvement registered for %s → Staging", model_id)

def trigger_ab_tests(**ctx):
    pass  # AB test registration handled inside HPOOptimizer._maybe_register_staging()

def clear_retrain_flags(**ctx):
    models = ctx["ti"].xcom_pull(key="models_to_optimize", task_ids="check_retrain_flags") or []
    for model_id in models:
        var = RETRAIN_FLAGS.get(model_id)
        if var:
            Variable.set(var, "false")
    logger.info("Cleared RETRAIN flags for: %s", models)

def write_hpo_summary(**ctx):
    run_ds = ctx["ds"]
    yyyymm = run_ds[:7].replace("-", "")
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    out = HPO_DIR / f"hpo_summary_{yyyymm}.md"
    ti  = ctx["ti"]
    rows = ""
    for model_id in RETRAIN_FLAGS:
        r = ti.xcom_pull(key=f"hpo_{model_id}", task_ids=f"optimize_all_models.optimize_{model_id}")
        if r:
            rows += f"| `{model_id}` | {r.get('best_metric','—')} | {r.get('best_value',0):.4f} | {r.get('n_trials_completed',0)} | {'✅' if r.get('promoted') else '—'} |\n"
    out.write_text(f"""# ANALYTICA HPO Summary — {run_ds[:7]}

> DAG: `{DAG_ID}`

| Model | Metric | Best Value | Trials | Auto-Staged |
|-------|--------|-----------|--------|-------------|
{rows if rows else '| — | No results | — | — | — |'}

*Cross-agent: VISUALIA reads this file for HPO dashboard update.*
""")

def emit_hpo_metrics(**ctx):
    ti = ctx["ti"]
    try:
        from src.data.metrics import HPO_BEST_METRIC, HPO_TRIALS_COMPLETED
        for model_id in RETRAIN_FLAGS:
            r = ti.xcom_pull(key=f"hpo_{model_id}", task_ids=f"optimize_all_models.optimize_{model_id}")
            if r:
                HPO_BEST_METRIC.labels(model_id=model_id, metric_name=r.get("best_metric","mae")).set(r.get("best_value",0))
                HPO_TRIALS_COMPLETED.labels(model_id=model_id).inc(r.get("n_trials_completed",0))
    except ImportError:
        pass

default_args = {"owner":"analytica","retries":1,"retry_delay":timedelta(minutes=10),
                "sla":timedelta(minutes=90),"email_on_failure":True}

with DAG(dag_id=DAG_ID, schedule_interval="0 2 10 * *", start_date=days_ago(1),
         default_args=default_args, catchup=False, max_active_runs=1,
         tags=["analytica","hpo","retraining"],
         description="Monthly HPO (Optuna TPE) for all 4 model families; auto-stages improvements and triggers A/B tests") as dag:

    t_flags = PythonOperator(task_id="check_retrain_flags", python_callable=check_retrain_flags)

    with TaskGroup("optimize_all_models") as tg:
        tasks = [PythonOperator(task_id=f"optimize_{m}", python_callable=_make_optimize(m))
                 for m in RETRAIN_FLAGS]
        for i in range(len(tasks) - 1):   # sequential to avoid GPU contention
            tasks[i] >> tasks[i+1]

    t_reg    = PythonOperator(task_id="register_improved_models", python_callable=register_improved_models)
    t_ab     = PythonOperator(task_id="trigger_ab_tests",         python_callable=trigger_ab_tests)
    t_clear  = PythonOperator(task_id="clear_retrain_flags",      python_callable=clear_retrain_flags)
    t_report = PythonOperator(task_id="write_hpo_summary",        python_callable=write_hpo_summary)
    t_emit   = PythonOperator(task_id="emit_hpo_metrics",         python_callable=emit_hpo_metrics)

    t_flags >> tg >> t_reg >> t_ab >> t_clear >> t_report >> t_emit
