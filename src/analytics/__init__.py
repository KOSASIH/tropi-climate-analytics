"""
Tropi-Climate-Analytics analytics package.
ML models, feature engineering, MLflow tracking, and SHAP explainability.
"""

from .mlflow_setup import (
    setup_experiments,
    ModelTracker,
    register_model,
    promote_model,
    get_production_model,
    list_model_versions,
    ModelRegistry,
)
from .feature_engineering import (
    FeatureMatrix,
    compute_rolling_stats,
    compute_lag_features,
    add_calendar_features,
    merge_gpm_bmkg,
    compute_spi,
    compute_soil_wetness_index,
)
from .models import (
    PrecipNowcastXGB,
    SeasonalProphet,
    SatelliteClassifierCNN,
)
from .retraining_pipeline import (
    RetrainingPipeline,
    ABTestManager,
    ModelType,
    RETRAIN_SCHEDULES,
    run_scheduled_retrain,
)
from .explainability import (
    ModelExplainer,
    batch_explain_predictions,
)

__all__ = [
    # MLflow
    "setup_experiments", "ModelTracker", "register_model", "promote_model",
    "get_production_model", "list_model_versions", "ModelRegistry",
    # Feature engineering
    "FeatureMatrix", "compute_rolling_stats", "compute_lag_features",
    "add_calendar_features", "merge_gpm_bmkg", "compute_spi", "compute_soil_wetness_index",
    # Models
    "PrecipNowcastXGB", "SeasonalProphet", "SatelliteClassifierCNN",
    # Retraining
    "RetrainingPipeline", "ABTestManager", "ModelType", "RETRAIN_SCHEDULES",
    "run_scheduled_retrain",
    # Explainability
    "ModelExplainer", "batch_explain_predictions",
]

__version__ = "0.1.0"
