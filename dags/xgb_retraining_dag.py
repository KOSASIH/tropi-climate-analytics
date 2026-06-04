"""
XGBoost Precipitation Retraining DAG — ANALYTICA Sprint 6 I1
dag_id: analytica_xgb_retraining
Schedule: 0 2 * * 1 Asia/Jakarta (weekly Monday 02:00 WIB)
  Also triggerable: set Airflow Variable RETRAIN_XGB=true

SLA: 90 minutes

Pipeline:
  check_retrain_flag
    -> load_training_data   (FeatureStoreClient, 90-day window, 10 precip feature refs)
    -> train_xgb            (XGBoost, 5-fold CV, early stopping)
    -> evaluate_model       (MAE / RMSE on held-out 15%)
    -> register_if_improved (MLflowRegistry; promote if MAE < champion * 0.97)
    -> clear_retrain_flag
    -> update_inference_cache

Prometheus: RETRAINING_DURATION{model_id='xgb_precipitation', status=success|failed}
MLflow experiment: xgb_precipitation_retraining
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAG_ID      = "analytica_xgb_retraining"
SCHEDULE    = "0 2 * * 1"
TIMEZONE    = "Asia/Jakarta"
SLA_SECONDS = 90 * 60
MODEL_ID    = "xgb_precip_nowcast"
EXPERIMENT  = "xgb_precipitation_retraining"
VARIABLE    = "RETRAIN_XGB"

TRAINING_WINDOW_DAYS = 90
HOLDOUT_FRAC         = 0.15
CV_FOLDS             = 5
PROMO_MAE_THRESHOLD  = 0.97  # challenger must be < champion * 0.97

PRECIP_FEATURE_REFS: List[str] = [
    "station_precipitation:precip_6hr",
    "station_precipitation:precip_24hr",
    "station_precipitation:precip_72hr",
    "grid_cell_features:t2m_mean",
    "grid_cell_features:rh_mean",
    "grid_cell_features:wind_speed_mean",
    "grid_cell_features:cape_j_kg",
    "grid_cell_features:tcwv_kgm2",
    "grid_cell_features:z500_m",
    "grid_cell_features:vorticity_850",
]

XGB_PARAMS: Dict[str, Any] = {
    "objective":         "reg:squarederror",
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "max_depth":         6,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_jobs":            -1,
    "random_state":      42,
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def check_retrain_flag(**context) -> bool:
    """
    SHORT_CIRCUIT if RETRAIN_XGB != 'true' AND the run was not manually triggered.
    Returns True to continue, False to short-circuit.
    """
    flag    = Variable.get(VARIABLE, default_var="false").strip().lower()
    is_manual = context.get("dag_run") and context["dag_run"].external_trigger
    should_run = (flag == "true") or bool(is_manual)
    if not should_run:
        logger.info("RETRAIN_XGB=%s and not manually triggered — short-circuiting", flag)
    return should_run


def load_training_data(**context) -> None:
    """
    Pull 90-day historical feature window from FeatureStoreClient.
    Entities: station_id + grid_cell_id.
    Applies FeatureStoreDataQuality gate before pushing to XCom.
    """
    from src.data.feature_store_client import FeatureStoreClient
    from src.data.data_quality         import FeatureStoreDataQuality, DataQualityError

    run_ds = context["ds"]
    end_dt = datetime.strptime(run_ds, "%Y-%m-%d")
    start  = (end_dt - timedelta(days=TRAINING_WINDOW_DAYS)).date()
    end    = end_dt.date()

    fs = FeatureStoreClient()
    df = fs.get_historical_features(
        feature_refs=PRECIP_FEATURE_REFS,
        start_date=start,
        end_date=end,
        entity_types=["station_id", "grid_cell_id"],
    )
    logger.info("Loaded %d rows × %d cols for training window %s→%s", len(df), len(df.columns), start, end)

    # Data quality gate — station_id suite
    dq = FeatureStoreDataQuality()
    try:
        dq.run_suite("station_id", df, run_date=end)
    except DataQualityError as exc:
        logger.error("Data quality gate blocked XGB training: %s", exc)
        raise

    # Persist training data to workspace (avoids XCom size limits)
    data_path = Path("workspace/data/training") / f"xgb_{run_ds.replace('-', '')}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    context["ti"].xcom_push(key="data_path", value=str(data_path))
    context["ti"].xcom_push(key="n_rows",    value=len(df))


def train_xgb(**context) -> None:
    """
    Train XGBoost with 5-fold CV + early stopping.
    Saves model artifact and CV results to workspace.
    """
    import mlflow
    import mlflow.xgboost
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics         import mean_absolute_error, mean_squared_error
    import xgboost as xgb

    ti        = context["ti"]
    data_path = ti.xcom_pull(key="data_path", task_ids="load_training_data")
    run_ds    = context["ds"]

    df       = pd.read_parquet(data_path)
    target   = "precip_6hr_actual" if "precip_6hr_actual" in df.columns else df.columns[-1]
    features = [c for c in df.columns if c != target and "timestamp" not in c.lower()]

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    # Chronological holdout split (no shuffle — preserves temporal integrity)
    split_idx    = int(len(X) * (1 - HOLDOUT_FRAC))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"xgb_train_{run_ds}") as run:
        mlflow.log_params({**XGB_PARAMS, "cv_folds": CV_FOLDS, "n_train": len(X_train)})

        # 5-fold CV on training split
        cv_model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
        kf       = KFold(n_splits=CV_FOLDS, shuffle=False)
        cv_maes  = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            cv_model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False,
            )
            val_pred = cv_model.predict(X_train[val_idx])
            cv_maes.append(mean_absolute_error(y_train[val_idx], val_pred))
            logger.debug("CV fold %d MAE: %.4f", fold + 1, cv_maes[-1])

        cv_mae_mean = float(np.mean(cv_maes))
        cv_mae_std  = float(np.std(cv_maes))
        mlflow.log_metric("cv_mae_mean", cv_mae_mean)
        mlflow.log_metric("cv_mae_std",  cv_mae_std)

        # Final model on full training split
        final_model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Holdout evaluation
        test_pred = final_model.predict(X_test)
        test_mae  = float(mean_absolute_error(y_test, test_pred))
        test_rmse = float(mean_squared_error(y_test, test_pred, squared=False))
        mlflow.log_metric("val_mae",  test_mae)
        mlflow.log_metric("val_rmse", test_rmse)
        logger.info("XGB holdout — MAE=%.4f RMSE=%.4f CV_MAE_mean=%.4f", test_mae, test_rmse, cv_mae_mean)

        # Log model artifact
        mlflow.xgboost.log_model(final_model, "model", input_example=X_test[:3])
        run_id = run.info.run_id

    ti.xcom_push(key="run_id",    value=run_id)
    ti.xcom_push(key="val_mae",   value=test_mae)
    ti.xcom_push(key="val_rmse",  value=test_rmse)
    ti.xcom_push(key="features",  value=features)


def evaluate_model(**context) -> None:
    """Log evaluation summary; gating logic lives in register_if_improved."""
    ti      = context["ti"]
    val_mae = ti.xcom_pull(key="val_mae",  task_ids="train_xgb")
    val_rmse = ti.xcom_pull(key="val_rmse", task_ids="train_xgb")
    logger.info("XGB evaluation: MAE=%.4f RMSE=%.4f", val_mae, val_rmse)
    ti.xcom_push(key="eval_ok", value=True)


def register_if_improved(**context) -> None:
    """
    Register new model version; promote to Production if MAE < champion_mae * 0.97.
    Tracks RETRAINING_DURATION Prometheus metric.
    """
    import mlflow
    from src.training.mlflow_registry import MLflowRegistry

    try:
        from src.data.metrics import RETRAINING_DURATION
        _metrics = True
    except ImportError:
        _metrics = False

    ti         = context["ti"]
    run_id     = ti.xcom_pull(key="run_id",  task_ids="train_xgb")
    val_mae    = ti.xcom_pull(key="val_mae", task_ids="train_xgb")
    val_rmse   = ti.xcom_pull(key="val_rmse", task_ids="train_xgb")

    t_start   = time.time()
    registry  = MLflowRegistry()
    status    = "failed"

    try:
        # Get current champion MAE
        champion_mae: Optional[float] = None
        try:
            champion_mv  = registry.get_latest_production(MODEL_ID)
            if champion_mv:
                client      = mlflow.MlflowClient()
                champ_run   = client.get_run(champion_mv.run_id)
                champion_mae = champ_run.data.metrics.get("val_mae")
        except Exception:
            pass

        new_mv = registry.register_model(
            run_id=run_id,
            model_name=MODEL_ID,
            metrics={"val_mae": val_mae, "val_rmse": val_rmse},
        )
        logger.info("Registered %s version %s (val_mae=%.4f)", MODEL_ID, new_mv.version, val_mae)

        promoted = False
        if champion_mae and val_mae < champion_mae * PROMO_MAE_THRESHOLD:
            registry.promote_to_production(MODEL_ID, new_mv.version)
            promoted = True
            logger.info(
                "PROMOTED %s v%s to Production (%.4f < %.4f * 0.97)",
                MODEL_ID, new_mv.version, val_mae, champion_mae,
            )
        elif champion_mae:
            logger.info(
                "NOT promoted: %.4f >= %.4f * 0.97 (%.4f)", val_mae, champion_mae, champion_mae * PROMO_MAE_THRESHOLD
            )
        else:
            registry.promote_to_production(MODEL_ID, new_mv.version)
            promoted = True
            logger.info("No champion found — promoting %s v%s as first Production", MODEL_ID, new_mv.version)

        status = "success"
        ti.xcom_push(key="promoted", value=promoted)
        ti.xcom_push(key="new_version", value=new_mv.version)
    except Exception as exc:
        logger.exception("register_if_improved failed: %s", exc)
        raise
    finally:
        duration = time.time() - t_start
        if _metrics:
            RETRAINING_DURATION.labels(model_id="xgb_precipitation", status=status).observe(duration)


def clear_retrain_flag(**context) -> None:
    """Reset RETRAIN_XGB Airflow Variable to 'false'."""
    Variable.set(VARIABLE, "false")
    logger.info("Cleared Airflow Variable %s", VARIABLE)


def update_inference_cache(**context) -> None:
    """Invalidate inference cache entries for the newly promoted model."""
    try:
        from src.serving.inference_cache import InferenceCache
        cache = InferenceCache()
        invalidated = cache.invalidate_model(MODEL_ID)
        logger.info("Cache invalidated for %s: %d entries cleared", MODEL_ID, invalidated)
    except Exception as exc:
        logger.warning("Cache invalidation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

default_args = {
    "owner":            "analytica",
    "depends_on_past":  False,
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "sla":              timedelta(seconds=SLA_SECONDS),
}

with DAG(
    dag_id=DAG_ID,
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["analytica", "retraining", "xgboost"],
    description="Weekly XGBoost precipitation nowcast retraining (RETRAIN_XGB-gated)",
) as dag:

    t_flag    = ShortCircuitOperator(task_id="check_retrain_flag",    python_callable=check_retrain_flag)
    t_load    = PythonOperator(     task_id="load_training_data",     python_callable=load_training_data)
    t_train   = PythonOperator(     task_id="train_xgb",              python_callable=train_xgb)
    t_eval    = PythonOperator(     task_id="evaluate_model",         python_callable=evaluate_model)
    t_reg     = PythonOperator(     task_id="register_if_improved",   python_callable=register_if_improved)
    t_clear   = PythonOperator(     task_id="clear_retrain_flag",     python_callable=clear_retrain_flag)
    t_cache   = PythonOperator(     task_id="update_inference_cache", python_callable=update_inference_cache)

    t_flag >> t_load >> t_train >> t_eval >> t_reg >> t_clear >> t_cache
