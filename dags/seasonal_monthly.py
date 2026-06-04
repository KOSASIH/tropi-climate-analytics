# ============================================================
# DAG: seasonal_monthly
# Water Availability Index (WAI) Computation
# Schedule: 1st of month, 04:00 WIB = 21:00 UTC (prev day)
# Owner: HYDROLOGIS | Infra: CLOUD-FORGE
# ============================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# Agricultural planning zones for WAI output
AGRI_ZONES = ["java", "sumatra", "kalimantan", "sulawesi", "nusa_tenggara"]

DEFAULT_ARGS = {
    "owner":            "hydrologis",
    "depends_on_past":  False,
    "email":            ["alerts@tropi-climate.id"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(hours=1),
    "execution_timeout": timedelta(hours=4),
}


@dag(
    dag_id="seasonal_monthly",
    description="Monthly Water Availability Index + 3-month seasonal forecast for agricultural planning",
    schedule_interval="0 21 1 * *",   # 21:00 UTC on last day of prev month = 04:00 WIB on 1st
    start_date=days_ago(31),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["hydrology", "seasonal", "water-availability", "monthly", "agricultural"],
    doc_md="""
    ## Seasonal Water Availability Index DAG
    Runs after `grace_monthly` to integrate groundwater storage with surface
    water balance (precipitation, ET, runoff) and produce a Water Availability
    Index (WAI) for each of Indonesia’s agricultural planning zones.

    Outputs a 3-month forecast using Prophet seasonal decomposition for
    irrigation advisory and BNPB drought risk pre-positioning.

    **Pipeline class**: `src/hydrology/seasonal/SeasonalWaterAvailabilityPipeline`

    **Outputs**:
    - WAI raster (1km) → S3 `processed-data/wai/YYYY/MM/`
    - 3-month forecast CSV → S3 + PostGIS `seasonal.water_availability_forecast`
    - Drought risk advisory → Kafka topic `seasonal.water_availability`
    - Government bulletin JSON → VISUALIA / TERRA-VISION dashboards
    """,
)
def seasonal_monthly_dag():

    # Depend on grace_monthly completing in the same month slot
    wait_for_grace = ExternalTaskSensor(
        task_id="wait_for_grace_monthly",
        external_dag_id="grace_monthly",
        external_task_id="store_and_publish_groundwater",
        execution_delta=timedelta(hours=1),   # grace runs 1h earlier (03:00 WIB)
        timeout=3600,
        poke_interval=120,
        mode="reschedule",
        allowed_states=["success", "skipped"],  # skipped = no new GRACE release
        failed_states=["failed"],
    )

    @task(task_id="aggregate_monthly_water_balance")
    def aggregate_water_balance(**context) -> dict:
        """
        Aggregate monthly water balance components:
        precipitation (GPM), actual ET (MODIS MOD16A2),
        surface runoff (VIC), soil storage (SMAP).
        """
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.seasonal import SeasonalWaterAvailabilityPipeline

        execution_dt = context["execution_date"]
        pipeline = SeasonalWaterAvailabilityPipeline(
            db_url=Variable.get("DATABASE_URL"),
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
        )
        balance = pipeline.aggregate_water_balance(
            target_month=execution_dt.replace(day=1) - timedelta(days=1),  # prev month
            components=["precipitation", "actual_et", "runoff", "soil_storage", "groundwater"],
        )
        log.info("Water balance aggregated for %d provinces", balance["province_count"])
        return balance

    @task(task_id="compute_water_availability_index")
    def compute_wai(balance: dict, **context) -> dict:
        """Compute Water Availability Index = (P + ΔStorage) / ET_potential per zone."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.seasonal import SeasonalWaterAvailabilityPipeline

        pipeline = SeasonalWaterAvailabilityPipeline(
            db_url=Variable.get("DATABASE_URL"),
        )
        wai = pipeline.compute_wai(
            water_balance=balance,
            normalization_period_years=30,   # 1991-2020 WMO normal
        )
        log.info(
            "WAI computed: deficit zones=%d, surplus zones=%d",
            wai["deficit_zone_count"], wai["surplus_zone_count"],
        )
        return wai

    @task(task_id="generate_3month_seasonal_forecast")
    def seasonal_forecast(wai: dict, **context) -> dict:
        """
        Use Prophet + ENSO/IOD teleconnection indices to produce
        3-month ahead WAI forecast per agricultural zone.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.seasonal import SeasonalWaterAvailabilityPipeline

        execution_dt = context["execution_date"]
        pipeline = SeasonalWaterAvailabilityPipeline(
            db_url=Variable.get("DATABASE_URL"),
            mlflow_tracking_uri=Variable.get("MLFLOW_TRACKING_URI"),
        )
        forecast = pipeline.run_seasonal_forecast(
            wai=wai,
            forecast_months=3,
            include_enso=True,
            include_iod=True,
            execution_dt=execution_dt,
        )
        log.info(
            "Seasonal forecast: %d zones, horizon=3 months, "
            "drought_risk_zones=%d",
            len(forecast["zones"]),
            forecast["drought_risk_count"],
        )
        return forecast

    @task.expand(
        task_id="generate_zone_advisory",
    )
    def zone_advisory(zone: str, forecast: dict, **context) -> dict:
        """Generate irrigation advisory per agricultural zone (fan-out)."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.seasonal import SeasonalWaterAvailabilityPipeline

        pipeline = SeasonalWaterAvailabilityPipeline()
        advisory = pipeline.generate_advisory(
            zone=zone,
            forecast=forecast,
        )
        log.info("[%s] Advisory: %s", zone.upper(), advisory["summary"])
        return advisory

    @task(task_id="publish_wai_and_advisories")
    def publish_results(wai: dict, forecast: dict, advisories: list[dict], **context) -> None:
        """Write to PostGIS, upload to S3, publish Kafka, emit dashboard payload."""
        import sys
        sys.path.insert(0, "/opt/airflow/src")
        from hydrology.seasonal import SeasonalWaterAvailabilityPipeline

        pipeline = SeasonalWaterAvailabilityPipeline(
            db_url=Variable.get("DATABASE_URL"),
            s3_bucket=Variable.get("S3_PROCESSED_DATA_BUCKET"),
            kafka_brokers=Variable.get("KAFKA_BOOTSTRAP_BROKERS"),
        )
        pipeline.publish(
            wai=wai,
            forecast=forecast,
            advisories=advisories,
            postgis_table="seasonal.water_availability_forecast",
            kafka_topic="seasonal.water_availability",
        )
        log.info("Seasonal WAI pipeline complete — dashboard payload dispatched")

    # ── Task graph ──────────────────────────────────────────
    balance    = aggregate_water_balance()
    wai        = compute_wai(balance)
    forecast   = seasonal_forecast(wai)
    advisories = zone_advisory.partial(forecast=forecast).expand(zone=AGRI_ZONES)

    wait_for_grace >> balance
    publish_results(wai, forecast, advisories)


seasonal_monthly_dag_instance = seasonal_monthly_dag()
