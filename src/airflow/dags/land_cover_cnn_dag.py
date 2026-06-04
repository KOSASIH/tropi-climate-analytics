"""
DAG: analytica_retrain_land_cover_cnn
Schedule: Quarterly — 1st of Jan / Apr / Jul / Oct at 03:00 WIB (Asia/Jakarta)
Cron:     0 3 1 */3 *

Wires RetrainingPipeline("land_cover_cnn") to Airflow PythonOperators.
Stages: validate_data → detect_drift → train_challenger → ab_evaluate
        → promote_or_archive → generate_model_card → notify_climate_os
"""

from airflow import DAG  # noqa: F401  — required for Airflow DAG discovery

from analytica.airflow.dag_factory import build_retraining_dag  # type: ignore[import]

dag: DAG = build_retraining_dag(
    model_type         = "land_cover_cnn",
    schedule_interval  = "0 3 1 */3 *",        # quarterly, 1st of month 03:00
    timezone           = "Asia/Jakarta",
    description        = (
        "ANALYTICA — Quarterly ResNet+Attention CNN land-cover retraining. "
        "Runs on 1 Jan / 1 Apr / 1 Jul / 1 Oct at 03:00 WIB. "
        "Ingests latest Landsat 8/9 OLI scenes from KLHK/LAPAN."
    ),
)
