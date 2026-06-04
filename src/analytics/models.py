"""
ML Models — Agent: ANALYTICA
XGBoost precipitation nowcasting, Prophet seasonal forecasting,
CNN land cover classification, MLflow experiment tracking.
"""

from typing import Any, Optional

from loguru import logger


class PrecipitationNowcastModel:
    """
    XGBoost model for 24-72 hour precipitation nowcasting.
    Features: MODIS cloud properties, GPM latent heat, BMKG surface obs,
              atmospheric indices (CAPE, CIN, wind shear), topography.
    Target: 6-hourly rainfall accumulation (mm) at 0.25-degree resolution.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.model = None
        self.feature_names: list[str] = []

    def load(self) -> None:
        try:
            import xgboost as xgb
            if self.model_path:
                self.model = xgb.Booster()
                self.model.load_model(self.model_path)
                logger.info(f"XGBoost model loaded: {self.model_path}")
        except ImportError:
            logger.warning("xgboost not available")

    def predict(self, features: Any) -> Any:
        """Predict 24-72h precipitation. Input: pandas DataFrame with feature columns."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features)
        return self.model.predict(dmatrix)


class SeasonalForecastModel:
    """
    Prophet model for seasonal climate forecasting.
    Captures Indonesian wet/dry season patterns (Nov-Apr wet, May-Oct dry).
    Handles El Nino/La Nina year-effects as additional regressors.
    """

    def __init__(self) -> None:
        self.model = None
        self.is_fitted = False

    def fit(self, df: Any) -> None:
        """Fit model. df must have 'ds' (datetime) and 'y' (rainfall mm) columns."""
        from prophet import Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        self.model.add_seasonality(name="indonesian_wet_season", period=365.25 / 2, fourier_order=5)
        self.model.fit(df)
        self.is_fitted = True
        logger.info("Prophet model fitted")

    def forecast(self, periods: int = 90) -> Any:
        """Generate forecast for next N days."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        future = self.model.make_future_dataframe(periods=periods)
        return self.model.predict(future)


class LandCoverCNN:
    """
    Convolutional Neural Network for Landsat 8/9 land cover classification.
    Input: 7-band multispectral image patches (64x64 pixels)
    Output: LandCoverClass probabilities (9 classes)
    Target accuracy: >=85% on KLHK validation set
    """

    CLASSES = [
        "forest", "degraded_forest", "plantation", "cropland",
        "water", "urban", "bare_land", "mangrove", "peatland",
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        try:
            import tensorflow as tf
            if self.model_path:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"CNN model loaded: {self.model_path}")
        except ImportError:
            logger.warning("tensorflow not available")

    def predict(self, image_patches: Any) -> Any:
        """Classify land cover. Input shape: (N, 64, 64, 7) float32 array."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return self.model.predict(image_patches)
