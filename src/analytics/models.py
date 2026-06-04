"""
ML model classes for Tropi-Climate-Analytics.
XGBoost (precipitation nowcasting), Prophet (seasonal), CNN (satellite classification).
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── XGBoost Precipitation Nowcasting ──────────────────────────────────────────

class PrecipNowcastXGB:
    """
    XGBoost ensemble for 24/48/72-hour precipitation nowcasting.
    One model per horizon trained on QPE-fused + SMAP + synoptic features.
    Target: total precipitation (mm) over the forecast horizon.
    """

    HORIZONS_H = [24, 48, 72]

    DEFAULT_PARAMS = {
        "n_estimators":      800,
        "learning_rate":     0.05,
        "max_depth":         7,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  5,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "objective":         "reg:squarederror",
        "eval_metric":       ["rmse", "mae"],
        "tree_method":       "hist",
        "device":            "cpu",
        "random_state":      42,
    }

    def __init__(self, params: Optional[Dict] = None):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("xgboost is required: pip install xgboost")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.models: Dict[int, Any] = {}
        self.feature_importances_: Dict[int, pd.Series] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_dict: Dict[int, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val_dict: Optional[Dict[int, pd.Series]] = None,
        early_stopping_rounds: int = 50,
    ) -> "PrecipNowcastXGB":
        """Train one model per horizon. y_dict keys must match HORIZONS_H."""
        for h in self.HORIZONS_H:
            if h not in y_dict:
                logger.warning("No target for horizon %dh, skipping", h)
                continue

            p = {**self.params}
            p.pop("random_state", None)
            model = self.xgb.XGBRegressor(**p, random_state=42)

            eval_set = [(X_train, y_dict[h])]
            if X_val is not None and y_val_dict and h in y_val_dict:
                eval_set.append((X_val, y_val_dict[h]))

            model.fit(
                X_train, y_dict[h],
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=100,
            )
            self.models[h] = model
            self.feature_importances_[h] = pd.Series(
                model.feature_importances_,
                index=X_train.columns,
            ).sort_values(ascending=False)
            logger.info("XGB h=%dh trained | best_iter=%d", h, model.best_iteration)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with columns precip_24h, precip_48h, precip_72h."""
        preds = {}
        for h, model in self.models.items():
            raw = model.predict(X)
            preds[f"precip_{h}h"] = np.clip(raw, 0, None)
        return pd.DataFrame(preds, index=X.index)

    def score(self, X: pd.DataFrame, y_dict: Dict[int, pd.Series]) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        metrics = {}
        preds = self.predict(X)
        for h in self.HORIZONS_H:
            if h not in y_dict:
                continue
            y_true = y_dict[h].reindex(X.index)
            y_pred = preds[f"precip_{h}h"]
            valid  = ~y_true.isna()
            rmse = mean_squared_error(y_true[valid], y_pred[valid], squared=False)
            mae  = mean_absolute_error(y_true[valid], y_pred[valid])
            metrics[f"rmse_{h}h"] = float(rmse)
            metrics[f"mae_{h}h"]  = float(mae)
        return metrics


# ── Prophet Seasonal Forecasting ──────────────────────────────────────────────

class SeasonalProphet:
    """
    Facebook Prophet wrapper for seasonal climate variable forecasting.
    Handles: temperature, monthly rainfall, drought index, water availability.
    Configured for tropical Indonesian seasonality (wet/dry, ENSO regressors).
    """

    DEFAULT_PARAMS = {
        "yearly_seasonality":  True,
        "weekly_seasonality":  False,
        "daily_seasonality":   False,
        "seasonality_mode":    "multiplicative",
        "changepoint_prior_scale":    0.05,
        "seasonality_prior_scale":    10.0,
        "interval_width":             0.90,
    }

    ENSO_REGRESSORS = ["oni_index", "iod_index", "mjo_phase_sin", "mjo_phase_cos"]

    def __init__(self, params: Optional[Dict] = None, add_enso: bool = True):
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            raise ImportError("prophet is required: pip install prophet")

        self.params    = {**self.DEFAULT_PARAMS, **(params or {})}
        self.add_enso  = add_enso
        self.models: Dict[str, Any] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}

    def fit(
        self,
        variable_series: Dict[str, pd.DataFrame],
        regressors_df: Optional[pd.DataFrame] = None,
    ) -> "SeasonalProphet":
        """
        variable_series: {variable_name: DataFrame with columns ['ds', 'y']}.
        regressors_df: optional DataFrame with columns ['ds'] + ENSO_REGRESSORS.
        """
        for var, df in variable_series.items():
            m = self.Prophet(**self.params)

            # Add dry/wet season custom seasonality (period ≈ 182.5 days)
            m.add_seasonality(name="wet_dry", period=182.5, fourier_order=5)

            if self.add_enso and regressors_df is not None:
                for reg in self.ENSO_REGRESSORS:
                    if reg in regressors_df.columns:
                        m.add_regressor(reg, standardize=True)
                train = df.merge(regressors_df[["ds"] + self.ENSO_REGRESSORS],
                                 on="ds", how="left")
            else:
                train = df.copy()

            m.fit(train)
            self.models[var] = m
            logger.info("Prophet fitted for '%s' (%d observations)", var, len(train))

        return self

    def predict(
        self,
        horizon_days: int = 365,
        regressors_future: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for var, m in self.models.items():
            future = m.make_future_dataframe(periods=horizon_days, freq="D")
            if self.add_enso and regressors_future is not None:
                future = future.merge(regressors_future[["ds"] + self.ENSO_REGRESSORS],
                                      on="ds", how="left").fillna(method="ffill")
            fc = m.predict(future)
            results[var] = fc[["ds", "yhat", "yhat_lower", "yhat_upper",
                                "trend", "yearly", "wet_dry"]]
            self.forecasts[var] = results[var]
        return results


# ── CNN Satellite Classification ──────────────────────────────────────────────

class SatelliteClassifierCNN:
    """
    Lightweight CNN for Landsat-8 / MODIS patch classification.
    Tasks: land_cover (10 classes), cloud_mask (binary), damage_assessment (3 classes).
    Input: (B, C, H, W) float32 patch tensors, pixel-normalised 0–1.
    """

    LAND_COVER_CLASSES = [
        "water", "urban", "cropland", "grassland", "shrubland",
        "broadleaf_forest", "needleleaf_forest", "mangrove",
        "bare_soil", "cloud_shadow",
    ]

    DAMAGE_CLASSES = ["no_damage", "partial_damage", "severe_damage"]

    def __init__(
        self,
        task: str = "land_cover",
        in_channels: int = 7,
        patch_size: int = 64,
        pretrained_path: Optional[str] = None,
    ):
        assert task in ("land_cover", "cloud_mask", "damage_assessment")
        self.task         = task
        self.in_channels  = in_channels
        self.patch_size   = patch_size
        self._model       = None
        self._build_model()
        if pretrained_path:
            self.load_weights(pretrained_path)

    def _build_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
        except ImportError:
            raise ImportError("PyTorch is required: pip install torch")

        n_classes = {
            "land_cover":        len(self.LAND_COVER_CLASSES),
            "cloud_mask":        2,
            "damage_assessment": len(self.DAMAGE_CLASSES),
        }[self.task]

        nn_ = self.torch.nn

        self._model = nn_.Sequential(
            # Block 1
            nn_.Conv2d(self.in_channels, 32, 3, padding=1), nn_.BatchNorm2d(32), nn_.ReLU(),
            nn_.Conv2d(32, 32, 3, padding=1),               nn_.BatchNorm2d(32), nn_.ReLU(),
            nn_.MaxPool2d(2), nn_.Dropout2d(0.1),

            # Block 2
            nn_.Conv2d(32, 64, 3, padding=1),  nn_.BatchNorm2d(64), nn_.ReLU(),
            nn_.Conv2d(64, 64, 3, padding=1),  nn_.BatchNorm2d(64), nn_.ReLU(),
            nn_.MaxPool2d(2), nn_.Dropout2d(0.1),

            # Block 3
            nn_.Conv2d(64, 128, 3, padding=1), nn_.BatchNorm2d(128), nn_.ReLU(),
            nn_.Conv2d(128, 128, 3, padding=1),nn_.BatchNorm2d(128), nn_.ReLU(),
            nn_.AdaptiveAvgPool2d((4, 4)),

            # Classifier head
            nn_.Flatten(),
            nn_.Linear(128 * 4 * 4, 256), nn_.ReLU(), nn_.Dropout(0.4),
            nn_.Linear(256, n_classes),
        )

    def predict_proba(self, patches: "torch.Tensor") -> "torch.Tensor":
        self._model.eval()
        with self.torch.no_grad():
            logits = self._model(patches)
            return self.torch.softmax(logits, dim=1)

    def predict(self, patches: "torch.Tensor") -> "torch.Tensor":
        return self.predict_proba(patches).argmax(dim=1)

    def load_weights(self, path: str) -> None:
        state = self.torch.load(path, map_location="cpu")
        self._model.load_state_dict(state)
        logger.info("Loaded CNN weights from %s", path)

    def save_weights(self, path: str) -> None:
        self.torch.save(self._model.state_dict(), path)

    @property
    def class_names(self) -> List[str]:
        return {
            "land_cover":        self.LAND_COVER_CLASSES,
            "cloud_mask":        ["clear", "cloud"],
            "damage_assessment": self.DAMAGE_CLASSES,
        }[self.task]
