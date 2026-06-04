# ============================================================
# DAG: grace_monthly
# GRACE-FO Groundwater Anomaly
# Schedule: 1st of month, 03:00 WIB = 20:00 UTC (prev day)
# Owner: HYDROLOGIS | Infra: CLOUD-FORGE
# ============================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "email":            ["alerts@tropi-climate.id"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(hours=1),
    "execution_timeout": timedelta(hours=3),
}


def _check_new_grace_release(**context) -> bool:
    """
    GRACE-FO monthly data typically released ~30 days after end of month.
    Short-circuit if no new release is detected to avoid re-processing.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from hydrology.groundwater import GRACEFOGroundwaterPipeline

    execution_dt = context["execution_date"]
    pipeline = GRACEFOGroundwaterPipeline(
        nasa_token=Variable.get("NASA_EARTHDATA_TOKEN"),
    )
    # Check if a new RL06 or RL07 mascon solution is available
    has_new = pipeline.check_new_release(execution_dt)
    log.info("GRACE-FO new release available: %s", has_new)
    return has_new


@dag(
    dag_id="grace_monthly",
    description="GRACE-FO monthly groundwater anomaly — aquifer depletion monitoring for Indonesia",
    schedule_interval="0 20 1 * *",   # 20:00 UTC on 1st = 03:00 WIB on 1st (+7)
    start_date=days_ago(31),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["hydrology", "grace-fo", "groundwater", "monthly", "nasa"],
    doc_md="""
    ## GRACE-FO Groundwater Anomaly DAG
    Ingests GRACE-FO RL07 mascon solutions (GSFC or CSR) to derive terrestrial
    water storage (TWS) anomalies, then applies GLDAS land surface model outputs
    to isolate groundwater storage change for Indonesian aquifer systems.

    **GRACE product**: GRCTellus.JPL.RL06.1Mascon.v03 or RL07
    **Pipeline class**: `src/hydrology/groundwater/GRACEFOGroundwaterPipeline`

    **Outputs**:
    - TWS anomaly GeoTIFF → S3 `processed-data/grace/YYYY/MM/`
    - Groundwater depletion raster → PostGIS `groundwater.monthly_anomaly`
    - Aquifer risk index per basin → Kafka topic `groundwater.monthly`
    """,
)
def grace_monthly_dag():

    check_release = ShortCircuitOperator(
        task_id="check_grace_fo_release",
        python_callable=_check_new_grace_release,
        ignore_downstream_trigger_rules=True,
    )

    @task(task_id="download_grace_mascon")
    def download_grace(**context) -> dict:
        """Download latest GRACE-FO mascon solution from NASA Earthdata."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.groundwater import GRACEFOGroundwaterPipeline

        execution_dt = context["execution_date"]
        pipeline = GRACEFOGroundwaterPipeline(
            nasa_token=Variable.get("NASA_EARTHDATA_TOKEN"),
            s3_bucket=Variable.get("S3_SATELLITE_DATA_BUCKET"),
        )
        result = pipeline.download_mascon(
            product="GRCTellus.JPL.RL06.1Mascon.v03",
            execution_dt=execution_dt,
        )
        log.info("GRACE mascon downloaded: %s", result["s3_key"])
        return result

    @task(task_id="compute_tws_anomaly")
    def compute_tws(download_result: dict, **context) -> dict:
        """Compute Terrestrial Water Storage (TWS) anomaly from GRACE mascon."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.groundwater import GRACEFOGroundwaterPipeline

        execution_dt = context["execution_date"]
        pipeline = GRACEFOGroundwaterPipeline(
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
            db_url=Variable.get("DATABASE_URL"),
        )
        tws = pipeline.compute_tws_anomaly(
            mascon_s3_key=download_result["s3_key"],
            baseline_period=("2004-01-01", "2009-12-31"),  # GRACE standard baseline
            aoi_bbox=[94.0, -11.0, 141.5, 6.5],
        )
        log.info("TWS anomaly range: %.2f to %.2f cm", tws["min_cm"], tws["max_cm"])
        return tws

    @task(task_id="isolate_groundwater_component")
    def isolate_groundwater(tws_result: dict, **context) -> dict:
        """
        Subtract soil moisture + snow water + surface water (from GLDAS)
        to isolate groundwater storage change (GWSC).
        """
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.groundwater import GRACEFOGroundwaterPipeline

        pipeline = GRACEFOGroundwaterPipeline(
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
            db_url=Variable.get("DATABASE_URL"),
        )
        gwsc = pipeline.isolate_groundwater(
            tws_result=tws_result,
            gldas_product="GLDAS_NOAH025_M.2.1",
        )
        log.info("Groundwater storage change: mean=%.2f cm, depletion_basins=%d",
                 gwsc["mean_cm"], gwsc["depletion_basin_count"])
        return gwsc

    @task(task_id="generate_aquifer_risk_index")
    def aquifer_risk(gwsc_result: dict, **context) -> dict:
        """Classify aquifer depletion risk per basin and generate alert levels."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.groundwater import GRACEFOGroundwaterPipeline

        pipeline = GRACEFOGroundwaterPipeline(
            db_url=Variable.get("DATABASE_URL"),
        )
        risk = pipeline.compute_aquifer_risk(
            gwsc_result=gwsc_result,
            trend_years=5,
        )
        critical = sum(1 for b in risk["basins"] if b["risk_level"] == "critical")
        log.info("Aquifer risk: %d basins critical, %d basins total",
                 critical, len(risk["basins"]))
        return risk

    @task(task_id="store_and_publish_groundwater")
    def store_and_publish(gwsc_result: dict, risk_index: dict, **context) -> None:
        """Write to PostGIS and publish to Kafka for downstream consumers."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.groundwater import GRACEFOGroundwaterPipeline

        pipeline = GRACEFOGroundwaterPipeline(
            db_url=Variable.get("DATABASE_URL"),
            kafka_brokers=Variable.get("KAFKA_BOOTSTRAP_BROKERS"),
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
        )
        pipeline.store_and_publish(
            gwsc_result=gwsc_result,
            risk_index=risk_index,
            postgis_table="groundwater.monthly_anomaly",
            kafka_topic="groundwater.monthly",
        )
        log.info("GRACE-FO monthly pipeline complete")

    # ── Task graph ──────────────────────────────────────────
    downloaded = download_grace()
    tws        = compute_tws(downloaded)
    gwsc       = isolate_groundwater(tws)
    risk       = aquifer_risk(gwsc)
    store_and_publish(gwsc, risk)

    check_release >> downloaded


grace_monthly_dag_instance = grace_monthly_dag()
