# ============================================================
# DAG: smap_daily
# SMAP L3 Soil Moisture Ingestion
# Schedule: Daily 06:00 WIB = 23:00 UTC
# Owner: HYDROLOGIS | Infra: CLOUD-FORGE
# ============================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "email":            ["alerts@tropi-climate.id"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=30),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id="smap_daily",
    description="Daily SMAP L3 Enhanced 9km soil moisture ingestion — drought risk input for HYDROLOGIS",
    schedule_interval="0 23 * * *",   # 23:00 UTC = 06:00 WIB (+7)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["hydrology", "smap", "soil-moisture", "daily", "nasa"],
    doc_md="""
    ## SMAP Daily Soil Moisture Ingestion
    Downloads SMAP L3 Enhanced 9km resolution daily composite from NASA
    Earthdata (SPL3SMP_E) for the previous UTC day, reprojects to Indonesia
    AOI (WGS84 ±10° lat, 94–142° lon), and loads into PostGIS + S3.

    **SMAP product**: SPL3SMP_E v006
    **Pipeline class**: `src/hydrology/soil_moisture/SMAPIngestionPipeline`

    **Outputs**:
    - HDF5 → S3 `satellite-data/smap/YYYY/MM/DD/`
    - NetCDF reprojected → S3 `processed-data/soil_moisture/YYYY/MM/DD/`
    - PostGIS table `soil_moisture.smap_daily`
    - Drought risk score → Kafka topic `soil_moisture.daily`
    """,
)
def smap_daily_dag():

    @task(task_id="check_smap_availability")
    def check_availability(**context) -> str:
        """Query NASA CMR to confirm SPL3SMP_E granule is available for target date."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.soil_moisture import SMAPIngestionPipeline

        # SMAP L3 daily composite is typically available 2-3 days after sensing
        # Run processes data for D-2 (yesterday's UTC) to ensure availability
        execution_dt = context["execution_date"]
        target_date = (execution_dt - timedelta(days=2)).date()

        pipeline = SMAPIngestionPipeline(
            nasa_token=Variable.get("NASA_EARTHDATA_TOKEN"),
        )
        granule_id = pipeline.check_granule_availability(
            product="SPL3SMP_E",
            version="006",
            target_date=target_date,
        )
        log.info("SMAP granule available: %s for date %s", granule_id, target_date)
        return granule_id

    @task(task_id="download_smap_hdf5")
    def download_smap(granule_id: str, **context) -> dict:
        """Download SMAP HDF5 granule from NASA Earthdata to S3."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.soil_moisture import SMAPIngestionPipeline

        pipeline = SMAPIngestionPipeline(
            nasa_token=Variable.get("NASA_EARTHDATA_TOKEN"),
            s3_bucket=Variable.get("S3_SATELLITE_DATA_BUCKET"),
        )
        result = pipeline.download_granule(granule_id)
        log.info("Downloaded SMAP granule → s3://%s/%s", result["bucket"], result["key"])
        return result

    @task(task_id="reproject_and_clip_aoi")
    def reproject_clip(download_result: dict, **context) -> dict:
        """Reproject SMAP EASE-Grid to WGS84, clip to Indonesia AOI, convert to NetCDF."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.soil_moisture import SMAPIngestionPipeline

        pipeline = SMAPIngestionPipeline(
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
        )
        result = pipeline.reproject_and_clip(
            s3_key=download_result["key"],
            aoi_bbox=[94.0, -11.0, 141.5, 6.5],   # Indonesia AOI
            target_crs="EPSG:4326",
            output_resolution_km=9,
        )
        log.info("Reprojected SMAP: %d valid pixels in Indonesia AOI", result["valid_pixels"])
        return result

    @task(task_id="compute_drought_risk_index")
    def compute_drought_risk(processed: dict, **context) -> dict:
        """Compute Soil Water Index (SWI) and drought risk score per watershed."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.soil_moisture import SMAPIngestionPipeline

        pipeline = SMAPIngestionPipeline(
            db_url=Variable.get("DATABASE_URL"),
        )
        risk_scores = pipeline.compute_drought_risk(
            netcdf_path=processed["s3_key"],
            historical_percentile_table="soil_moisture.historical_percentiles",
        )
        drought_alert_count = sum(1 for s in risk_scores["watersheds"] if s["risk_level"] >= 3)
        log.info("Drought risk computed: %d watersheds, %d with risk >= 3",
                 len(risk_scores["watersheds"]), drought_alert_count)
        return risk_scores

    @task(task_id="load_to_postgis_and_publish")
    def load_and_publish(processed: dict, risk_scores: dict, **context) -> None:
        """Write to PostGIS soil_moisture.smap_daily and publish to Kafka."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.soil_moisture import SMAPIngestionPipeline

        pipeline = SMAPIngestionPipeline(
            db_url=Variable.get("DATABASE_URL"),
            kafka_brokers=Variable.get("KAFKA_BOOTSTRAP_BROKERS"),
        )
        pipeline.load_to_postgis(processed)
        pipeline.publish_to_kafka(
            risk_scores=risk_scores,
            topic="soil_moisture.daily",
        )
        log.info("SMAP daily pipeline complete — data loaded and published")

    # ── Task graph ──────────────────────────────────────────
    granule_id  = check_availability()
    downloaded  = download_smap(granule_id)
    processed   = reproject_clip(downloaded)
    risk_scores = compute_drought_risk(processed)
    load_and_publish(processed, risk_scores)


smap_daily_dag_instance = smap_daily_dag()
