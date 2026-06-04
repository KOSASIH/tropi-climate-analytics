"""
ML Models — Agent: ANALYTICA
XGBoost precipitation nowcasting, Prophet seasonal forecasting,
CNN land cover classification + MLflow tracking and SHAP explainability.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .mlflow_setup import setup_mlflow


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Precipitation Nowcasting
# ─────────────────────────────────────────────────────────────────────────────

class PrecipitationNowcastModel:
    """
    XGBoost model for 24-72 hour precipitation nowcasting.
    Features: MODIS cloud properties, GPM latent heat, BMKG surface obs,
              atmospheric indices (CAPE, CIN, wind shear), topography.
    Target  : 6-hourly rainfall accumulation (mm) at 0.25-degree resolution.
    """

    FEATURE_GROUPS: Dict[str, List[str]] = {
        "modis":           ["cloud_optical_depth", "cloud_effective_radius",
                            "cloud_top_temp", "cloud_fraction", "cloud_water_path",
                            "aerosol_optical_depth", "land_surface_temp", "ndvi", "evi"],
        "gpm":             ["precip_rate_1h", "precip_rate_3h", "precip_rate_6h",
                            "latent_heat_flux", "precip_probability", "precip_type"],
        "bmkg":            ["station_rainfall", "temp_2m", "dewpoint_2m", "rh_2m",
                            "wind_speed_10m", "wind_dir_10m", "sea_level_pressure",
                            "station_count_50km"],
        "atmospheric":     ["cape", "cin", "lifted_index", "wind_shear_0_6km",
                            "precipitable_water_total", "k_index", "totals_totals"],
        "topography":      ["elevation", "slope", "aspect",
                            "distance_to_coast_km", "terrain_roughness_index"],
        "temporal":        ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                            "month_sin", "month_cos", "is_wet_season"],
        "climate_indices": ["oni", "iod", "enso_phase", "mjo_phase", "mjo_amplitude"],
        "spatial_lags":    [f"precip_lag_{h}h" for h in [6, 12, 18, 24, 48, 72]],
    }

    XGBOOST_PARAMS: Dict[str, Any] = {
        "objective":             "reg:squarederror",
        "eval_metric":           ["rmse", "mae"],
        "n_estimators":          1000,
        "max_depth":             8,
        "learning_rate":         0.05,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "min_child_weight":      5,
        "reg_alpha":             0.1,
        "reg_lambda":            1.0,
        "tree_method":           "hist",
        "early_stopping_rounds": 50,
        "n_jobs":                -1,
        "random_state":          42,
    }

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path     = model_path
        self.model          = None
        self.feature_names: List[str] = []
        self.shap_explainer = None
        self.version:       Optional[str] = None
        self.metrics:       Dict[str, float] = {}

    @property
    def all_feature_names(self) -> List[str]:
        return [f for grp in self.FEATURE_GROUPS.values() for f in grp]

    def load(self) -> None:
        try:
            import xgboost as xgb
            if self.model_path:
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_path)
                logger.info(f"XGBoost model loaded: {self.model_path}")
        except ImportError:
            logger.warning("xgboost not available")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train with full MLflow tracking. Returns eval metrics."""
        import mlflow
        import mlflow.xgboost
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        setup_mlflow()
        run_name = run_name or f"xgboost_nowcast_{datetime.utcnow():%Y%m%d_%H%M%S}"
        self.feature_names = list(X_train.columns)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_type", "xgboost_precipitation_nowcast")
            mlflow.set_tag("region",     "Indonesia")
            mlflow.log_params(self.XGBOOST_PARAMS)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_features",      X_train.shape[1])

            self.model = xgb.XGBRegressor(**self.XGBOOST_PARAMS)
            self.model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)], verbose=100)

            y_pred = self.model.predict(X_val)
            rmse   = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae    = float(mean_absolute_error(y_val, y_pred))
            bias   = float(np.mean(y_pred - y_val))

            self.metrics = {"val_rmse": rmse, "val_mae": mae, "val_bias": bias}
            mlflow.log_metrics(self.metrics)

            fi  = dict(zip(self.feature_names, self.model.feature_importances_))
            top = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20])
            mlflow.log_dict(top, "top_20_feature_importances.json")

            mlflow.xgboost.log_model(
                self.model, "model",
                registered_model_name="tropi-precipitation-nowcast",
            )
            self.version = run.info.run_id

        logger.info(f"XGBoost done | RMSE={rmse:.3f} mm | MAE={mae:.3f} mm")
        return self.metrics

    def predict(self, features: Any) -> Any:
        """Predict 24-72 h precipitation. Input: pandas DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.predict(features)

    def explain(self, features: pd.DataFrame, max_display: int = 20) -> Dict[str, Any]:
        """SHAP TreeExplainer - feature attribution per prediction."""
        import shap
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.model)
        sv       = self.shap_explainer.shap_values(features)
        mean_abs = dict(zip(self.feature_names, np.abs(sv).mean(axis=0)))
        return {
            "shap_values":        sv,
            "expected_value":     self.shap_explainer.expected_value,
            "feature_attribution": mean_abs,
            "top_features":       dict(sorted(mean_abs.items(),
                                              key=lambda x: x[1], reverse=True)[:max_display]),
        }

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_model(path)
            logger.info(f"XGBoost model saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Prophet Seasonal Forecasting
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalForecastModel:
    """
    Prophet model for seasonal climate forecasting.
    Captures Indonesian wet/dry season patterns (Nov-Apr wet, May-Oct dry).
    El Nino/La Nina handled via ONI, IOD, and ENSO-phase regressors.
    """

    WET_SEASON_MONTHS  = {11, 12, 1, 2, 3, 4}
    CLIMATE_REGRESSORS = ["oni", "iod", "enso_phase", "mjo_phase", "mjo_amplitude"]

    def __init__(self) -> None:
        self.model:     Any                  = None
        self.is_fitted: bool                 = False
        self.version:   Optional[str]        = None
        self.metrics:   Dict[str, float]     = {}

    def _build_model(self) -> Any:
        from prophet import Prophet
        m = Prophet(
            yearly_seasonality      = True,
            weekly_seasonality      = False,
            daily_seasonality       = False,
            seasonality_mode        = "multiplicative",
            changepoint_prior_scale = 0.05,
            seasonality_prior_scale = 10.0,
            interval_width          = 0.95,
        )
        # Indonesian semi-annual wet season
        m.add_seasonality(name="indonesian_wet_season",
                          period=365.25 / 2, fourier_order=5)
        # ENSO-modulated annual cycle
        m.add_seasonality(name="enso_annual", period=365.25,
                          fourier_order=8, condition_name="is_enso_year")
        for reg in self.CLIMATE_REGRESSORS:
            m.add_regressor(reg, standardize=True)
        return m

    @classmethod
    def _add_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["is_enso_year"]  = (df.get("enso_phase", 0) != 0).astype(float)
        df["is_wet_season"] = df["ds"].dt.month.isin(cls.WET_SEASON_MONTHS).astype(float)
        for reg in cls.CLIMATE_REGRESSORS:
            if reg not in df.columns:
                df[reg] = 0.0
        return df

    def fit(self, df: Any, run_name: Optional[str] = None) -> None:
        """
        Fit model. df must have 'ds' (datetime) and 'y' (rainfall mm) columns,
        plus optional climate regressor columns.
        """
        import mlflow
        setup_mlflow()
        run_name = run_name or f"prophet_seasonal_{datetime.utcnow():%Y%m%d_%H%M%S}"
        df = self._add_features(df)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_type", "prophet_seasonal_forecast")
            mlflow.set_tag("region",     "Indonesia")
            mlflow.log_param("seasonality_mode", "multiplicative")
            mlflow.log_param("regressors",       self.CLIMATE_REGRESSORS)
            mlflow.log_param("n_train_samples",  len(df))

            self.model     = self._build_model()
            self.model.fit(df)
            self.is_fitted = True
            self.version   = run.info.run_id

        logger.info(f"Prophet fitted | run_id={self.version}")

    def forecast(self, periods: int = 90,
                 future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecast for next N days (or over supplied future_df)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        if future_df is None:
            future = self.model.make_future_dataframe(periods=periods)
            for reg in self.CLIMATE_REGRESSORS:
                future[reg] = 0.0
            future = self._add_features(future)
        else:
            future = self._add_features(future_df)

        fc   = self.model.predict(future)
        keep = ["ds", "yhat", "yhat_lower", "yhat_upper",
                "trend", "indonesian_wet_season"]
        return fc[[c for c in keep if c in fc.columns]]

    def cross_validate(self, horizon: str = "90 days",
                       initial: str = "730 days",
                       period: str  = "180 days") -> Dict[str, float]:
        from prophet.diagnostics import cross_validation, performance_metrics
        import mlflow
        df_cv   = cross_validation(self.model, initial=initial,
                                   period=period, horizon=horizon)
        df_perf = performance_metrics(df_cv)
        self.metrics = {
            "cv_rmse": float(df_perf["rmse"].mean()),
            "cv_mae":  float(df_perf["mae"].mean()),
            "cv_mape": float(df_perf["mape"].mean()),
        }
        mlflow.log_metrics(self.metrics)
        logger.info(f"Prophet CV | RMSE={self.metrics['cv_rmse']:.3f} mm")
        return self.metrics

    def explain(self, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        components = {}
        for col in (["trend", "yearly", "indonesian_wet_season", "enso_annual"]
                    + self.CLIMATE_REGRESSORS):
            if col in forecast_df.columns:
                components[col] = forecast_df[col].tolist()
        return {
            "method":     "prophet_decomposition",
            "components": components,
            "note":       ("Multiplicative decomposition. Each component shows its "
                           "contribution to the forecast relative to the trend."),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CNN Land Cover Classifier
# ─────────────────────────────────────────────────────────────────────────────

class LandCoverCNN:
    """
    Convolutional Neural Network for Landsat 8/9 land cover classification.
    Input : 7-band multispectral image patches (64x64 pixels)
    Output: LandCoverClass probabilities (9 classes)
    Target: >= 85% accuracy on KLHK validation set
    """

    CLASSES = [
        "forest", "degraded_forest", "plantation", "cropland",
        "water", "urban", "bare_land", "mangrove", "peatland",
    ]
    NUM_CLASSES = 9
    PATCH_SIZE  = 64
    NUM_BANDS   = 7

    TRAIN_PARAMS: Dict[str, Any] = {
        "batch_size":               64,
        "epochs":                   100,
        "learning_rate":            1e-3,
        "dropout_rate":             0.3,
        "l2_reg":                   1e-4,
        "early_stopping_patience":  15,
        "reduce_lr_patience":       5,
    }

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path     = model_path
        self.model          = None
        self.shap_explainer = None
        self.version:       Optional[str]    = None
        self.metrics:       Dict[str, float] = {}

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build_architecture(self) -> Any:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers
        L2 = regularizers.l2(self.TRAIN_PARAMS["l2_reg"])

        inp = layers.Input(shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.NUM_BANDS))
        x   = layers.RandomFlip("horizontal_and_vertical")(inp)
        x   = layers.RandomRotation(0.1)(x)
        x   = layers.Conv2D(32, 3, padding="same", activation="relu",
                            kernel_regularizer=L2)(x)
        x   = layers.BatchNormalization()(x)

        for filters in (64, 128, 256):
            x = self._res_block(x, filters, L2)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.2)(x)

        x = self._channel_attention(x, ratio=8)
        x = self._res_block(x, 512, L2)

        x   = layers.GlobalAveragePooling2D()(x)
        x   = layers.Dense(256, activation="relu", kernel_regularizer=L2)(x)
        x   = layers.Dropout(self.TRAIN_PARAMS["dropout_rate"])(x)
        out = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)

        return models.Model(inp, out, name="LandCoverCNN")

    @staticmethod
    def _res_block(x: Any, filters: int, l2: Any) -> Any:
        from tensorflow.keras import layers
        sc = x
        x  = layers.Conv2D(filters, 3, padding="same", activation="relu",
                           kernel_regularizer=l2)(x)
        x  = layers.BatchNormalization()(x)
        x  = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=l2)(x)
        x  = layers.BatchNormalization()(x)
        if sc.shape[-1] != filters:
            sc = layers.Conv2D(filters, 1, padding="same")(sc)
        x = layers.Add()([sc, x])
        return layers.Activation("relu")(x)

    @staticmethod
    def _channel_attention(x: Any, ratio: int = 8) -> Any:
        from tensorflow.keras import layers
        c   = x.shape[-1]
        gap = layers.GlobalAveragePooling2D()(x)
        gmp = layers.GlobalMaxPooling2D()(x)
        gap = layers.Dense(c // ratio, activation="relu")(gap)
        gmp = layers.Dense(c // ratio, activation="relu")(gmp)
        gap = layers.Dense(c, activation="sigmoid")(gap)
        gmp = layers.Dense(c, activation="sigmoid")(gmp)
        sc  = layers.Add()([gap, gmp])
        sc  = layers.Reshape((1, 1, c))(sc)
        return layers.Multiply()([x, sc])

    # ── Training / inference ─────────────────────────────────────────────────

    def load(self) -> None:
        try:
            import tensorflow as tf
            if self.model_path:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"CNN model loaded: {self.model_path}")
        except ImportError:
            logger.warning("tensorflow not available")

    def train(
        self,
        X_train: Any, y_train: Any,
        X_val:   Any, y_val:   Any,
        run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        import mlflow
        import tensorflow as tf
        from tensorflow.keras import callbacks
        from sklearn.metrics import classification_report

        setup_mlflow()
        run_name = run_name or f"landcover_cnn_{datetime.utcnow():%Y%m%d_%H%M%S}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_type", "cnn_land_cover")
            mlflow.set_tag("region",     "Indonesia")
            mlflow.log_params(self.TRAIN_PARAMS)
            mlflow.log_param("num_classes", self.NUM_CLASSES)
            mlflow.log_param("classes",     self.CLASSES)

            self.model = self._build_architecture()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.TRAIN_PARAMS["learning_rate"]),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            Path("models").mkdir(exist_ok=True)
            cb_list = [
                callbacks.EarlyStopping(
                    patience=self.TRAIN_PARAMS["early_stopping_patience"],
                    restore_best_weights=True, monitor="val_accuracy"),
                callbacks.ReduceLROnPlateau(
                    patience=self.TRAIN_PARAMS["reduce_lr_patience"],
                    factor=0.5, monitor="val_loss"),
                callbacks.ModelCheckpoint(
                    "models/landcover_cnn_best.h5",
                    save_best_only=True, monitor="val_accuracy"),
            ]

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.TRAIN_PARAMS["batch_size"],
                epochs=self.TRAIN_PARAMS["epochs"],
                callbacks=cb_list, verbose=1,
            )

            y_pred = np.argmax(self.model.predict(X_val), axis=1)
            acc    = float(np.mean(y_pred == y_val))
            loss   = float(history.history["val_loss"][-1])

            self.metrics = {
                "val_accuracy":        acc,
                "val_loss":            loss,
                "target_accuracy_met": float(acc >= 0.85),
            }
            mlflow.log_metrics(self.metrics)

            report = classification_report(y_val, y_pred,
                                           target_names=self.CLASSES,
                                           output_dict=True)
            for cls, m in report.items():
                if isinstance(m, dict):
                    mlflow.log_metric(f"f1_{cls.replace(' ', '_')}",
                                      m.get("f1-score", 0))

            mlflow.tensorflow.log_model(
                self.model, "model",
                registered_model_name="tropi-land-cover-cnn",
            )
            self.version = run.info.run_id

        status = "PASS" if acc >= 0.85 else "FAIL"
        logger.info(f"CNN done | Accuracy={acc:.3%} | Target 85% [{status}]")
        return self.metrics

    def predict(self, image_patches: Any) -> Dict[str, Any]:
        """Classify land cover. Input: (N, 64, 64, 7) float32 array."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        probs   = self.model.predict(image_patches)
        indices = np.argmax(probs, axis=1)
        return {
            "class_indices": indices,
            "class_labels":  [self.CLASSES[i] for i in indices],
            "probabilities": probs,
        }

    def explain(self, image_patches: Any, n_background: int = 50) -> Dict[str, Any]:
        """SHAP GradientExplainer - per-band attribution maps."""
        import shap
        if self.shap_explainer is None:
            self.shap_explainer = shap.GradientExplainer(
                self.model, image_patches[:n_background])
        sv         = self.shap_explainer.shap_values(image_patches[:10])
        band_names = ["B1_coastal", "B2_blue", "B3_green",
                      "B4_red", "B5_nir", "B6_swir1", "B7_swir2"]
        band_imp = {
            band: float(np.abs(sv[..., i]).mean())
            for i, band in enumerate(band_names)
        }
        return {
            "method":          "shap_gradient_explainer",
            "band_importance": band_imp,
            "shap_shape":      list(np.array(sv).shape),
        }
