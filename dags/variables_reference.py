# ============================================================
# Airflow Variables Reference — Tropi Climate Analytics
# Required Variables for HYDROLOGIS DAGs
# Set via: airflow variables set KEY VALUE
# Or via: Airflow UI > Admin > Variables
# ============================================================

# These variables must be set in the Airflow deployment.
# Values should be sourced from AWS Secrets Manager.
# CLOUD-FORGE manages secrets population via deploy scripts.

VARIABLES = {
    # NASA Earthdata (all NASA satellite downloads)
    "NASA_EARTHDATA_TOKEN": "<from-secrets-manager: nasa/earthdata-token>",

    # BMKG API (gauge observations)
    "BMKG_API_KEY": "<from-secrets-manager: bmkg/api-key>",

    # Database (Aurora PostgreSQL + PostGIS)
    "DATABASE_URL": "postgresql://tropi_admin:<password>@<rds-writer-endpoint>:5432/tropi_climate",

    # Redis (ElastiCache, for antecedent state caching)
    "REDIS_URL": "rediss://<elasticache-primary-endpoint>:6379/0",

    # S3 Buckets
    "S3_SATELLITE_DATA_BUCKET": "tropi-climate-prod-satellite-data",
    "S3_PROCESSED_DATA_BUCKET": "tropi-climate-prod-processed-data",

    # MSK Kafka (TLS broker list)
    "KAFKA_BOOTSTRAP_BROKERS": "<from-outputs: msk.bootstrap_brokers_tls>",

    # MLflow (model registry for LSTM, Prophet)
    "MLFLOW_TRACKING_URI": "http://mlflow.tropi-climate.id",

    # BPBD Alert Webhook (optional: flood alert dispatch)
    "BPBD_ALERT_WEBHOOK_URL": "<from-secrets-manager: bpbd/webhook-url>",
}

# Kafka Topics produced by HYDROLOGIS DAGs
KAFKA_TOPICS = [
    "qpe.realtime",           # QPE fusion output (30min)
    "flood.early_warning",    # Flood forecast + BPBD alerts (30min)
    "soil_moisture.daily",    # SMAP drought risk (daily)
    "groundwater.monthly",    # GRACE-FO aquifer risk (monthly)
    "seasonal.water_availability",  # WAI + advisory (monthly)
]
