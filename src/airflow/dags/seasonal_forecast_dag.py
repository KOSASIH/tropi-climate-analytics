"""
DAG: analytica_retrain_seasonal_forecast
Schedule: Monthly — 1st of every month 02:00 WIB (Asia/Jakarta)
Cron:     0 2 1 * *

Wires RetrainingPipeline("seasonal_forecast") to Airflow PythonOperators.
Stages: validate_data → detect_drift → train_challenger → ab_evaluate
        → promote_or_archive → generate_model_card → notify_climate_os
"""

from airflow import DAG  # noqa: F401  — required for Airflow DAG discovery

from analytica.airflow.dag_factory import build_retraining_dag  # type: ignore[import]

dag: DAG = build_retraining_dag(
    model_type         = "seasonal_forecast",
    schedule_interval  = "0 2 1 * *",          # 1st of month 02:00
    timezone           = "Asia/Jakarta",
    description        = (
        "ANALYTICA — Monthly Prophet seasonal forecast retraining. "
        "Runs on the 1st of each month at 02:00 WIB. "
        "Incorporates updated ENSO/IOD/MJO climate index regressors."
    ),
)
