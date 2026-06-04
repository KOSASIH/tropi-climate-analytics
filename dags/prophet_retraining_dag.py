"""
Prophet Seasonal Retraining DAG — ANALYTICA Sprint 6 I2
dag_id: analytica_prophet_retraining
Schedule: 0 3 1 * * Asia/Jakarta (monthly 1st 03:00 WIB)
  Also triggerable: set Airflow Variable RETRAIN_PROPHET=true

SLA: 60 minutes

Pipeline:
  check_retrain_flag
    -> load_seasonal_data   (FeatureStoreClient, 24-month window, watershed_id entity)
    -> train_prophet        (trend + yearly + weekly seasonality, holidays=Indonesia)
    -> evaluate_model       (MAPE on last 90 days)
    -> register_if_improved (MLflowRegistry; promote if MAPE < champion * 0.97)
    -> clear_retrain_flag

Prometheus: RETRAINING_DURATION{model_id='prophet_seasonal', status=success|failed}
MLflow experiment: prophet_seasonal_retraining
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DAG_ID      = "analytica_prophet_retraining"
SCHEDULE    = "0 3 1 * *"
TIMEZONE    = "Asia/Jakarta"
SLA_SECONDS = 60 * 60
MODEL_ID    = "prophet_seasonal"
EXPERIMENT  = "prophet_seasonal_retraining"
VARIABLE    = "RETRAIN_PROPHET"

TRAINING_WINDOW_MONTHS = 24
EVAL_WINDOW_DAYS       = 90
PROMO_MAPE_THRESHOLD   = 0.97

SEASONAL_FEATURE_REFS: List[str] = [
    "watershed_features:streamflow_monthly_mean",
    "watershed_features:precip_monthly_total",
    "watershed_features:soil_moisture_monthly_mean",
    "watershed_features:evapotranspiration_monthly",
    "watershed_features:ndvi_monthly_mean",
]

# Indonesia public holidays (representative; extend via holiday library)
INDONESIA_HOLIDAYS = {
    "Tahun Baru":         "01-01",
    "Hari Kemerdekaan":   "08-17",
    "Hari Natal":         "12-25",
    "Hari Buruh":         "05-01",
    "Hari Pancasila":     "06-01",
}


def check_retrain_flag(**context) -> bool:
    flag      = Variable.get(VARIABLE, default_var="false").strip().lower()
    is_manual = context.get("dag_run") and context["dag_run"].external_trigger
    should_run = (flag == "true") or bool(is_manual)
    if not should_run:
        logger.info("RETRAIN_PROPHET=%s and not manually triggered — short-circuiting", flag)
    return should_run


def load_seasonal_data(**context) -> None:
    from src.data.feature_store_client import FeatureStoreClient
    from src.data.data_quality         import FeatureStoreDataQuality, DataQualityError

    run_ds = context["ds"]
    end_dt = datetime.strptime(run_ds, "%Y-%m-%d")
    start  = (end_dt - timedelta(days=TRAINING_WINDOW_MONTHS * 30)).date()
    end    = end_dt.date()

    fs = FeatureStoreClient()
    df = fs.get_historical_features(
        feature_refs=SEASONAL_FEATURE_REFS,
        start_date=start,
        end_date=end,
        entity_types=["watershed_id"],
    )
    logger.info("Prophet: loaded %d rows for %s→%s", len(df), start, end)

    dq = FeatureStoreDataQuality()
    try:
        dq.run_suite("watershed_id", df, run_date=end)
    except DataQualityError as exc:
        logger.error("DQ gate blocked Prophet training: %s", exc)
        raise

    data_path = Path("workspace/data/training") / f"prophet_{run_ds.replace('-', '')}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    context["ti"].xcom_push(key="data_path", value=str(data_path))


def train_prophet(**context) -> None:
    import mlflow
    import mlflow.pyfunc
    from prophet import Prophet

    ti        = context["ti"]
    data_path = ti.xcom_pull(key="data_path", task_ids="load_seasonal_data")
    run_ds    = context["ds"]

    df = pd.read_parquet(data_path)

    # Prophet requires ds + y columns
    ts_col = next((c for c in df.columns if "timestamp" in c.lower() or c == "ds"), None)
    y_col  = next((c for c in df.columns if "streamflow" in c.lower() or "precip" in c.lower()), df.columns[-1])

    if ts_col and ts_col != "ds":
        df = df.rename(columns={ts_col: "ds"})
    elif "ds" not in df.columns:
        df["ds"] = pd.date_range(end=run_ds, periods=len(df), freq="MS")

    df = df.rename(columns={y_col: "y"}).dropna(subset=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])

    # Build Indonesia holidays DataFrame
    years = df["ds"].dt.year.unique().tolist()
    holiday_rows = []
    for year in years:
        for h_name, mmdd in INDONESIA_HOLIDAYS.items():
            try:
                holiday_rows.append({"holiday": h_name, "ds": pd.Timestamp(f"{year}-{mmdd}"), "lower_window": 0, "upper_window": 1})
            except ValueError:
                pass
    holidays_df = pd.DataFrame(holiday_rows) if holiday_rows else None

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"prophet_train_{run_ds}") as run:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            holidays=holidays_df,
        )

        # Add monthly seasonality for tropical climate patterns
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        # Chronological split — last 90 days as eval holdout
        cutoff = df["ds"].max() - pd.Timedelta(days=EVAL_WINDOW_DAYS)
        train_df = df[df["ds"] <= cutoff]
        test_df  = df[df["ds"] >  cutoff]

        m.fit(train_df)

        # Evaluate MAPE on holdout
        future   = m.make_future_dataframe(periods=len(test_df), freq="MS")
        forecast = m.predict(future)
        pred_df  = forecast[forecast["ds"].isin(test_df["ds"])][["ds", "yhat"]].merge(test_df[["ds", "y"]], on="ds")
        mape     = float(np.mean(np.abs((pred_df["y"] - pred_df["yhat"]) / (pred_df["y"].abs() + 1e-8))))

        mlflow.log_metric("val_mape",  mape)
        mlflow.log_metric("n_train",   len(train_df))
        mlflow.log_metric("n_eval",    len(pred_df))
        logger.info("Prophet holdout MAPE=%.4f", mape)

        # Log model as pyfunc artifact
        import pickle, tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "prophet_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(m, f)
            mlflow.log_artifact(model_path, artifact_path="model")

        run_id = run.info.run_id

    ti.xcom_push(key="run_id",   value=run_id)
    ti.xcom_push(key="val_mape", value=mape)


def evaluate_model(**context) -> None:
    mape = context["ti"].xcom_pull(key="val_mape", task_ids="train_prophet")
    logger.info("Prophet evaluation: MAPE=%.4f", mape)


def register_if_improved(**context) -> None:
    import mlflow
    from src.training.mlflow_registry import MLflowRegistry

    try:
        from src.data.metrics import RETRAINING_DURATION
        _metrics = True
    except ImportError:
        _metrics = False

    ti       = context["ti"]
    run_id   = ti.xcom_pull(key="run_id",   task_ids="train_prophet")
    val_mape = ti.xcom_pull(key="val_mape", task_ids="train_prophet")
    t_start  = time.time()
    status   = "failed"
    registry = MLflowRegistry()

    try:
        champion_mape: Optional[float] = None
        try:
            mv = registry.get_latest_production(MODEL_ID)
            if mv:
                champ_metrics = mlflow.MlflowClient().get_run(mv.run_id).data.metrics
                champion_mape = champ_metrics.get("val_mape")
        except Exception:
            pass

        new_mv = registry.register_model(run_id=run_id, model_name=MODEL_ID, metrics={"val_mape": val_mape})

        if champion_mape and val_mape >= champion_mape * PROMO_MAPE_THRESHOLD:
            logger.info("NOT promoted: MAPE %.4f >= champion %.4f * 0.97", val_mape, champion_mape)
        else:
            registry.promote_to_production(MODEL_ID, new_mv.version)
            logger.info("PROMOTED %s v%s (MAPE=%.4f)", MODEL_ID, new_mv.version, val_mape)

        status = "success"
    except Exception as exc:
        logger.exception("register_if_improved(prophet) failed: %s", exc)
        raise
    finally:
        if _metrics:
            RETRAINING_DURATION.labels(model_id="prophet_seasonal", status=status).observe(time.time() - t_start)


def clear_retrain_flag(**context) -> None:
    Variable.set(VARIABLE, "false")
    logger.info("Cleared %s", VARIABLE)


default_args = {
    "owner": "analytica", "depends_on_past": False, "email_on_failure": True,
    "retries": 1, "retry_delay": timedelta(minutes=10),
    "sla": timedelta(seconds=SLA_SECONDS),
}

with DAG(
    dag_id=DAG_ID, schedule_interval=SCHEDULE, start_date=days_ago(1),
    default_args=default_args, catchup=False,
    tags=["analytica", "retraining", "prophet"],
    description="Monthly Prophet seasonal retraining (RETRAIN_PROPHET-gated)",
) as dag:
    t_flag  = ShortCircuitOperator(task_id="check_retrain_flag",  python_callable=check_retrain_flag)
    t_load  = PythonOperator(task_id="load_seasonal_data",        python_callable=load_seasonal_data)
    t_train = PythonOperator(task_id="train_prophet",             python_callable=train_prophet)
    t_eval  = PythonOperator(task_id="evaluate_model",            python_callable=evaluate_model)
    t_reg   = PythonOperator(task_id="register_if_improved",      python_callable=register_if_improved)
    t_clear = PythonOperator(task_id="clear_retrain_flag",        python_callable=clear_retrain_flag)

    t_flag >> t_load >> t_train >> t_eval >> t_reg >> t_clear
