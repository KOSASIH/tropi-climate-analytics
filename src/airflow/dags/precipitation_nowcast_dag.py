"""
DAG: analytica_retrain_precipitation_nowcast
Schedule: Weekly — every Monday 01:00 WIB (Asia/Jakarta)
Cron:     0 1 * * 1

Wires RetrainingPipeline("precipitation_nowcast") to Airflow PythonOperators.
Stages: validate_data → detect_drift → train_challenger → ab_evaluate
        → promote_or_archive → generate_model_card → notify_climate_os
"""

from airflow import DAG  # noqa: F401  — required for Airflow DAG discovery

from analytica.airflow.dag_factory import build_retraining_dag  # type: ignore[import]

dag: DAG = build_retraining_dag(
    model_type         = "precipitation_nowcast",
    schedule_interval  = "0 1 * * 1",          # every Monday 01:00
    timezone           = "Asia/Jakarta",
    description        = (
        "ANALYTICA — Weekly XGBoost precipitation nowcast retraining. "
        "Runs every Monday at 01:00 WIB. "
        "Drift threshold: PSI > 0.2 or KS p < 0.05."
    ),
)
