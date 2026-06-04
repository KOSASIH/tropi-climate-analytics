"""
MLflow Setup — ANALYTICA
Initialises the tracking server and creates the canonical experiment.
"""

import os
from loguru import logger

MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "tropi-climate-models")

MODEL_REGISTRY_NAMES = {
    "precipitation_nowcast": "tropi-precipitation-nowcast",
    "seasonal_forecast":     "tropi-seasonal-forecast",
    "land_cover_cnn":        "tropi-land-cover-cnn",
    "climate_transformer":   "tropi-climate-transformer",
}


def setup_mlflow() -> str:
    """
    Point MLflow at the tracking server, create (or re-use) the experiment,
    and return the experiment_id.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI -> {MLFLOW_TRACKING_URI}")

    try:
        experiment_id = mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            tags={
                "project":  "tropi-climate-analytics",
                "domain":   "climate_forecasting",
                "region":   "Indonesia",
                "agent":    "ANALYTICA",
            },
        )
        logger.info(f"MLflow experiment created: {MLFLOW_EXPERIMENT_NAME} (id={experiment_id})")
    except Exception:
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        experiment_id = exp.experiment_id
        logger.info(f"MLflow experiment exists: {MLFLOW_EXPERIMENT_NAME} (id={experiment_id})")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return experiment_id


def get_latest_model_version(model_name: str, stage: str = "Production"):
    """Return the latest registered model version in the given stage."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None
    except Exception as exc:
        logger.warning(f"Could not fetch model version for {model_name}: {exc}")
        return None


def promote_model(model_name: str, version: str, stage: str = "Production") -> None:
    """Promote a model version to a new stage."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage,
        archive_existing_versions=True,
    )
    logger.info(f"Promoted {model_name} v{version} -> {stage}")
