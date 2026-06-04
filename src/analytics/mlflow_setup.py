"""
MLflow tracking and model registry setup for Tropi-Climate-Analytics.
Configures experiments, artifact storage (S3), and model lifecycle management.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.tropi-climate-analytics.internal:5000"
)
MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET", "tropi-climate-mlflow-artifacts")
MLFLOW_S3_PREFIX = os.getenv("MLFLOW_S3_PREFIX", "mlflow")

EXPERIMENTS = {
    "precipitation_nowcasting": {
        "description": "XGBoost 24-72h precipitation nowcasting from GPM+BMKG fusion",
        "tags": {"domain": "hydrology", "data_source": "GPM,BMKG", "region": "Indonesia"},
    },
    "seasonal_forecasting": {
        "description": "Prophet seasonal climate forecasting (temperature, rainfall, drought index)",
        "tags": {"domain": "climate", "data_source": "BMKG,ERA5", "region": "Indonesia"},
    },
    "satellite_classification": {
        "description": "CNN land cover, cloud masking, and damage assessment on Landsat/MODIS",
        "tags": {"domain": "remote_sensing", "data_source": "Landsat8,MODIS", "resolution": "30m"},
    },
    "streamflow_forecasting": {
        "description": "LSTM multi-station streamflow forecasting (Ciliwung, Brantas, Solo)",
        "tags": {"domain": "hydrology", "data_source": "BMKG,SMAP,GPM", "model_type": "LSTM"},
    },
    "flood_early_warning": {
        "description": "Ensemble flood risk scoring for BPBD alert integration",
        "tags": {"domain": "disaster_risk", "output": "bpbd_webhook", "latency_target": "30min"},
    },
}

# ── Client setup ───────────────────────────────────────────────────────────────

def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def setup_experiments() -> Dict[str, str]:
    """Create or retrieve all platform experiments. Returns {name: experiment_id}."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = get_client()
    ids = {}
    for name, cfg in EXPERIMENTS.items():
        artifact_location = f"s3://{MLFLOW_S3_BUCKET}/{MLFLOW_S3_PREFIX}/{name}"
        existing = client.get_experiment_by_name(name)
        if existing is None:
            exp_id = client.create_experiment(
                name=name,
                artifact_location=artifact_location,
                tags=cfg["tags"],
            )
            logger.info("Created experiment '%s' (id=%s)", name, exp_id)
        else:
            exp_id = existing.experiment_id
            logger.debug("Experiment '%s' already exists (id=%s)", name, exp_id)
        ids[name] = exp_id
    return ids


# ── Run helpers ────────────────────────────────────────────────────────────────

class ModelTracker:
    """Context manager for an MLflow training run with auto-logging."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._run = None

    def __enter__(self) -> "ModelTracker":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        logger.info("MLflow run started: %s / %s", self.experiment_name, self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()
        return False  # re-raise exceptions

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str, input_example=None) -> str:
        """Log model and register it. Returns model URI."""
        signature = None
        if input_example is not None:
            try:
                preds = model.predict(input_example)
                signature = infer_signature(input_example, preds)
            except Exception:
                pass

        if hasattr(model, "save"):
            # Keras / TF
            mlflow.keras.log_model(model, artifact_path, signature=signature,
                                   input_example=input_example)
        elif hasattr(model, "get_booster"):
            # XGBoost
            mlflow.xgboost.log_model(model, artifact_path, signature=signature,
                                     input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, artifact_path, signature=signature,
                                     input_example=input_example)

        return f"runs:/{self.run_id}/{artifact_path}"

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)


# ── Registry helpers ───────────────────────────────────────────────────────────

def register_model(
    run_uri: str,
    model_name: str,
    description: Optional[str] = None,
) -> str:
    """Register a logged model and return the version."""
    result = mlflow.register_model(run_uri, model_name, await_registration_for=60)
    version = result.version
    client = get_client()
    if description:
        client.update_model_version(model_name, version, description=description)
    logger.info("Registered model '%s' version %s", model_name, version)
    return version


def promote_model(model_name: str, version: str, stage: str = "Production") -> None:
    """Transition a model version to Staging or Production."""
    client = get_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info("Promoted '%s' v%s → %s", model_name, version, stage)


def get_production_model(model_name: str) -> Any:
    """Load the current Production version of a registered model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)


def list_model_versions(model_name: str, stage: Optional[str] = None) -> list:
    client = get_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    if stage:
        versions = [v for v in versions if v.current_stage == stage]
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


# ── Registered model names (shared constants) ─────────────────────────────────

class ModelRegistry:
    PRECIP_NOWCAST_XGB   = "precipitation-nowcasting-xgboost"
    SEASONAL_PROPHET     = "seasonal-climate-prophet"
    LAND_COVER_CNN       = "land-cover-classification-cnn"
    CLOUD_MASK_CNN       = "cloud-masking-cnn"
    DAMAGE_ASSESS_CNN    = "damage-assessment-cnn"
    STREAMFLOW_LSTM      = "lstm-streamflow-v1"
    FLOOD_RISK_ENSEMBLE  = "flood-risk-ensemble"
