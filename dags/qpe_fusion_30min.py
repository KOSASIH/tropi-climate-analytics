# ============================================================
# DAG: qpe_fusion_30min
# QPE Fusion — GPM IMERG-HHR + BMKG Kriging
# Schedule: Every 30 minutes
# SLA: Results available within 20 min of data availability
# Owner: HYDROLOGIS | Infra: CLOUD-FORGE
# ============================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "email":            ["alerts@tropi-climate.id"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "sla":              timedelta(minutes=20),
    "execution_timeout": timedelta(minutes=25),
}


@dag(
    dag_id="qpe_fusion_30min",
    description="GPM IMERG-HHR + BMKG gauge Kriging QPE fusion — 4km resolution updated every 30 min",
    schedule_interval="*/30 * * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,            # prevent overlap on slow runs
    default_args=DEFAULT_ARGS,
    tags=["hydrology", "qpe", "gpm", "bmkg", "30min"],
    doc_md="""
    ## QPE Fusion DAG
    Fuses GPM IMERG half-hourly (HHR) satellite precipitation with BMKG
    rain-gauge observations using Kriging interpolation to produce a 4km
    resolution quantitative precipitation estimate (QPE) grid for Indonesia.

    **Pipeline class**: `src/hydrology/qpe/gpm_bmkg_fusion.GPMBMKGFusionPipeline`

    **Outputs**:
    - NetCDF4 QPE grid → S3 `processed-data/qpe/YYYY/MM/DD/HH{mm}/`
    - PostGIS table `qpe.half_hourly_estimates`
    - Kafka topic `qpe.realtime` (downstream flood DAG)
    """,
)
def qpe_fusion_dag():

    @task(task_id="fetch_gpm_imerg_hhr")
    def fetch_gpm_imerg(**context) -> dict:
        """Download GPM IMERG-HHR granule for the current half-hour window."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.qpe.gpm_bmkg_fusion import GPMBMKGFusionPipeline

        execution_dt = context["execution_date"]
        pipeline = GPMBMKGFusionPipeline(
            nasa_token=Variable.get("NASA_EARTHDATA_TOKEN"),
            s3_bucket=Variable.get("S3_SATELLITE_DATA_BUCKET"),
        )
        granule_path = pipeline.fetch_gpm_hhr(execution_dt)
        log.info("GPM HHR granule fetched: %s", granule_path)
        return {"granule_path": granule_path, "execution_dt": execution_dt.isoformat()}

    @task(task_id="fetch_bmkg_gauges")
    def fetch_bmkg_gauges(**context) -> dict:
        """Pull BMKG rain-gauge observations for the current 30-min window."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.qpe.gpm_bmkg_fusion import GPMBMKGFusionPipeline

        execution_dt = context["execution_date"]
        pipeline = GPMBMKGFusionPipeline(
            bmkg_api_key=Variable.get("BMKG_API_KEY"),
        )
        gauge_data = pipeline.fetch_bmkg_gauges(execution_dt)
        log.info("BMKG gauges fetched: %d stations", len(gauge_data))
        return {"gauge_count": len(gauge_data), "execution_dt": execution_dt.isoformat()}

    @task(task_id="kriging_fusion")
    def kriging_fusion(gpm_result: dict, bmkg_result: dict, **context) -> dict:
        """Merge GPM satellite grid with BMKG gauge Kriging interpolation."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.qpe.gpm_bmkg_fusion import GPMBMKGFusionPipeline

        execution_dt = context["execution_date"]
        pipeline = GPMBMKGFusionPipeline(
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
            db_url=Variable.get("DATABASE_URL"),
        )
        output = pipeline.run_kriging_fusion(
            granule_path=gpm_result["granule_path"],
            execution_dt=execution_dt,
        )
        log.info("Kriging fusion complete — RMSE: %.4f mm/h", output.get("rmse", 0))
        return output

    @task(task_id="store_qpe_output")
    def store_qpe_output(fusion_result: dict, **context) -> None:
        """Persist QPE NetCDF to S3 and write to PostGIS."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.qpe.gpm_bmkg_fusion import GPMBMKGFusionPipeline

        pipeline = GPMBMKGFusionPipeline(
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
            db_url=Variable.get("DATABASE_URL"),
            kafka_brokers=Variable.get("KAFKA_BOOTSTRAP_BROKERS"),
        )
        pipeline.store_and_publish(
            fusion_result=fusion_result,
            kafka_topic="qpe.realtime",
        )
        log.info("QPE stored and published to Kafka topic qpe.realtime")

    # ── Task graph ──────────────────────────────────────────
    gpm    = fetch_gpm_imerg()
    bmkg   = fetch_bmkg_gauges()
    fused  = kriging_fusion(gpm, bmkg)
    store_qpe_output(fused)


qpe_fusion_dag_instance = qpe_fusion_dag()
