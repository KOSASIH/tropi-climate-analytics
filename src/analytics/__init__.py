"""
ANALYTICA — src/analytics package
Sprint 0: MLflow setup, feature engineering, ML models, retraining, explainability.
"""

from .explainability import ModelCardGenerator, ModelExplainer
from .feature_engineering import FeatureEngineeringPipeline
from .mlflow_setup import (
    MODEL_REGISTRY_NAMES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    get_latest_model_version,
    promote_model,
    setup_mlflow,
)
from .models import LandCoverCNN, PrecipitationNowcastModel, SeasonalForecastModel
from .retraining_pipeline import (
    ChampionChallengerManager,
    DataDriftDetector,
    RETRAINING_SCHEDULE,
    RetrainingPipeline,
)

__all__ = [
    # MLflow
    "setup_mlflow",
    "get_latest_model_version",
    "promote_model",
    "MODEL_REGISTRY_NAMES",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    # Feature engineering
    "FeatureEngineeringPipeline",
    # Models
    "PrecipitationNowcastModel",
    "SeasonalForecastModel",
    "LandCoverCNN",
    # Retraining
    "DataDriftDetector",
    "ChampionChallengerManager",
    "RetrainingPipeline",
    "RETRAINING_SCHEDULE",
    # Explainability
    "ModelExplainer",
    "ModelCardGenerator",
]

__version__ = "0.1.0"
__sprint__  = "Sprint 0"
__agent__   = "ANALYTICA"
