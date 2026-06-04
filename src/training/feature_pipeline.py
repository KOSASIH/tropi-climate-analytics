"""
Feature Pipeline — ANALYTICA Sprint 7 L3
Module: src/training/feature_pipeline.py

Class: FeaturePipeline
  compute_features(entity_df, entity_type) → pd.DataFrame
  select_features(df, target, method='shap') → FeatureSelectionResult
  run_full_pipeline(entity_type, start_date, end_date) → pd.DataFrame

Feature engineering per entity_type:
  station_id:   lag features (1h,3h,6h,12h,24h,48h), rolling stats (mean/std/max),
                hour-of-day + day-of-week cyclical encoding (sin/cos), inter-station spatial lag
  watershed_id: monthly SPI/SPEI rolling, lag streamflow (6h,12h,24h,72h), upstream area ratio,
                antecedent soil moisture index (5-day weighted), seasonal dummy
  grid_cell_id: NDVI lag (8-day), T2M diurnal range, humidity deficit, orographic uplift proxy

Feature selection:
  shap:         SHAPExplainer on last 30d predictions → rank by mean |SHAP|, keep top-K=30
  rfecv:        sklearn RFECV with XGBoost estimator (5-fold, scoring=neg_MAE)
  mutual_info:  sklearn mutual_info_regression, keep features with MI > 0.01

Output: workspace/output/feature_pipeline/selected_features_{entity_type}_{YYYYMMDD}.json
Prometheus: FEATURE_IMPORTANCE_UPDATE{entity_type, method} Counter
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR    = Path("workspace/output/feature_pipeline")
EXPLAINER_DIR = Path("workspace/output/explainability")
SHAP_TOP_K    = 30

LAG_OFFSETS_STATION   = [1, 3, 6, 12, 24, 48]   # hours
LAG_OFFSETS_WATERSHED = [6, 12, 24, 72]           # hours
ROLLING_WINDOWS       = [3, 6, 12, 24]            # hours
ROLLING_STATS         = ["mean", "std", "max"]
SPI_WINDOWS           = [1, 3, 6]                 # months
AMI_WEIGHTS_5D        = [0.4, 0.25, 0.15, 0.1, 0.1]  # most recent first


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionResult:
    entity_type:       str
    method:            str
    n_in:              int
    n_out:             int
    selected_features: List[str]
    importance_scores: Dict[str, float]
    run_date:          str
    output_path:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "entity_type":       self.entity_type,
            "method":            self.method,
            "n_in":              self.n_in,
            "n_out":             self.n_out,
            "selected_features": self.selected_features,
            "importance_scores": self.importance_scores,
            "run_date":          self.run_date,
            "output_path":       self.output_path,
        }


# ---------------------------------------------------------------------------
# Cyclical encoding
# ---------------------------------------------------------------------------

def _sin_cos(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    return (
        np.sin(2 * np.pi * series / period).rename(f"{series.name}_sin"),
        np.cos(2 * np.pi * series / period).rename(f"{series.name}_cos"),
    )


# ---------------------------------------------------------------------------
# SPI stub (simplified; production would use scipy.stats.norm.ppf(ecdf))
# ---------------------------------------------------------------------------

def _spi(precip: pd.Series, window_months: int) -> pd.Series:
    rolled = precip.rolling(window=window_months, min_periods=1).mean()
    mean, std = rolled.mean(), rolled.std() + 1e-8
    z = (rolled - mean) / std
    return z.rename(f"spi_{window_months}m")


# ---------------------------------------------------------------------------
# Station-ID feature engineering
# ---------------------------------------------------------------------------

def _engineer_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for station_id entity (precipitation nowcasting).
    Expects: 'timestamp' column (hourly cadence), precip/RH/T2M/wind columns.
    """
    feats = df.copy()
    precip_col = next((c for c in df.columns if "precip" in c.lower()), None)
    ts_col     = next((c for c in df.columns if "timestamp" in c.lower() or c == "ds"), None)

    if ts_col:
        ts = pd.to_datetime(feats[ts_col])
        hod_sin, hod_cos = _sin_cos(ts.dt.hour, 24)
        dow_sin, dow_cos = _sin_cos(ts.dt.dayofweek, 7)
        feats["hour_sin"] = hod_sin.values
        feats["hour_cos"] = hod_cos.values
        feats["dow_sin"]  = dow_sin.values
        feats["dow_cos"]  = dow_cos.values

    for numeric_col in feats.select_dtypes(include=[np.number]).columns:
        # Lag features
        for lag in LAG_OFFSETS_STATION:
            feats[f"{numeric_col}_lag{lag}h"] = feats[numeric_col].shift(lag)
        # Rolling stats
        for win in ROLLING_WINDOWS:
            roll = feats[numeric_col].rolling(window=win, min_periods=1)
            for stat in ROLLING_STATS:
                feats[f"{numeric_col}_roll{win}h_{stat}"] = getattr(roll, stat)()

    # Inter-station spatial lag proxy: cumulative 24h precipitation momentum
    if precip_col:
        feats["precip_momentum_24h"] = feats[precip_col].rolling(24, min_periods=1).sum()
        feats["precip_cumulative_48h"] = feats[precip_col].rolling(48, min_periods=1).sum()

    logger.debug("station_id features: %d → %d", len(df.columns), len(feats.columns))
    return feats


# ---------------------------------------------------------------------------
# Watershed-ID feature engineering
# ---------------------------------------------------------------------------

def _engineer_watershed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for watershed_id entity (streamflow / seasonal forecasting).
    Expects: streamflow_cms, precip columns, soil_moisture; monthly cadence preferred.
    """
    feats = df.copy()
    ts_col  = next((c for c in df.columns if "timestamp" in c.lower() or c == "ds"), None)
    precip_col = next((c for c in df.columns if "precip" in c.lower()), None)
    sm_col     = next((c for c in df.columns if "soil_moisture" in c.lower()), None)
    sf_col     = next((c for c in df.columns if "streamflow" in c.lower()), None)

    # SPI/SPEI rolling
    if precip_col:
        for w in SPI_WINDOWS:
            feats[f"spi_{w}m"] = _spi(feats[precip_col], w).values

    # Lag streamflow
    if sf_col:
        for lag in LAG_OFFSETS_WATERSHED:
            feats[f"{sf_col}_lag{lag}h"] = feats[sf_col].shift(lag)

    # Antecedent Soil Moisture Index (AMI) — 5-day weighted sum
    if sm_col:
        weights = np.array(AMI_WEIGHTS_5D)
        ami_values = []
        sm_vals = feats[sm_col].fillna(0).values
        for i in range(len(sm_vals)):
            window = sm_vals[max(0, i - 4):i + 1][::-1]
            w_clip = weights[:len(window)]
            ami_values.append(float(np.dot(window, w_clip / w_clip.sum())))
        feats["ami_5d"] = ami_values

    # Upstream area ratio proxy (static per-watershed; would be joined from metadata)
    feats["upstream_area_ratio"] = 1.0  # placeholder; real values from watershed metadata

    # Seasonal dummy (Indonesian wet/dry season)
    if ts_col:
        ts = pd.to_datetime(feats[ts_col])
        feats["is_wet_season"] = ts.dt.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
        feats["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

    logger.debug("watershed_id features: %d → %d", len(df.columns), len(feats.columns))
    return feats


# ---------------------------------------------------------------------------
# Grid-cell-ID feature engineering
# ---------------------------------------------------------------------------

def _engineer_grid_cell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for grid_cell_id entity (CNN/XGB gridded features).
    Expects: T2M, RH, wind_speed, NDVI columns.
    """
    feats = df.copy()
    t2m_col    = next((c for c in df.columns if c.lower() in ("t2m", "temperature_2m", "t_2m")), None)
    rh_col     = next((c for c in df.columns if "rh" in c.lower() or "relative_humidity" in c.lower()), None)
    ndvi_col   = next((c for c in df.columns if "ndvi" in c.lower()), None)
    ts_col     = next((c for c in df.columns if "timestamp" in c.lower() or c == "ds"), None)

    # T2M diurnal range (daily max - min)
    if t2m_col:
        feats["t2m_diurnal_range"] = feats[t2m_col].rolling(24, min_periods=1).max() - \
                                     feats[t2m_col].rolling(24, min_periods=1).min()
        feats["t2m_lag24h"] = feats[t2m_col].shift(24)

    # Humidity deficit (saturation vapour pressure approximation)
    if t2m_col and rh_col:
        # Magnus formula: es(T) = 6.112 * exp(17.67*T/(T+243.5))
        es = 6.112 * np.exp(17.67 * feats[t2m_col] / (feats[t2m_col] + 243.5))
        e  = es * feats[rh_col] / 100.0
        feats["humidity_deficit"] = (es - e).clip(lower=0)

    # NDVI lag (8-day cadence → ~8 steps per change)
    if ndvi_col:
        feats["ndvi_lag8d"]  = feats[ndvi_col].shift(8)
        feats["ndvi_change"] = feats[ndvi_col] - feats["ndvi_lag8d"]

    # Orographic uplift proxy: elevation × wind speed (requires DEM join)
    wind_col = next((c for c in df.columns if "wind" in c.lower() and "speed" in c.lower()), None)
    if wind_col:
        elev_col = next((c for c in df.columns if "elev" in c.lower() or "dem" in c.lower()), None)
        if elev_col:
            feats["orographic_uplift"] = feats[elev_col] * feats[wind_col]
        else:
            feats["orographic_uplift_proxy"] = feats[wind_col]  # bare proxy without DEM

    logger.debug("grid_cell_id features: %d → %d", len(df.columns), len(feats.columns))
    return feats


_ENGINEER_MAP = {
    "station_id":   _engineer_station,
    "grid_cell_id": _engineer_grid_cell,
    "watershed_id": _engineer_watershed,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Automated feature engineering and selection for ANALYTICA training workflows.
    Called by retraining DAGs (I1–I4) load_training_data tasks.
    """

    def compute_features(self, entity_df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        """
        Apply entity-specific feature engineering to raw feature DataFrame.

        Args:
            entity_df:   Raw feature DataFrame from FeatureStoreClient.
            entity_type: 'station_id' | 'watershed_id' | 'grid_cell_id'

        Returns:
            Enriched feature DataFrame with all derived features.
        """
        if entity_type not in _ENGINEER_MAP:
            raise ValueError(f"Unknown entity_type '{entity_type}'. Valid: {list(_ENGINEER_MAP)}")
        engineered = _ENGINEER_MAP[entity_type](entity_df)
        # Drop columns with >30% NaN (likely entirely from lagging at the start of the window)
        thresh = int(0.7 * len(engineered))
        engineered = engineered.dropna(axis=1, thresh=thresh)
        return engineered

    def select_features(
        self,
        df:          pd.DataFrame,
        target:      str,
        method:      str = "shap",
        entity_type: str = "station_id",
        run_date:    Optional[date] = None,
    ) -> FeatureSelectionResult:
        """
        Select the best predictive features using the specified method.

        Args:
            df:          Feature DataFrame (including target column).
            target:      Name of the target column.
            method:      'shap' | 'rfecv' | 'mutual_info'
            entity_type: Used for output filename and Prometheus labels.
            run_date:    Date of this run.

        Returns:
            FeatureSelectionResult with selected features and importance scores.
        """
        run_date = run_date or date.today()
        feat_cols = [c for c in df.columns if c != target and "timestamp" not in c.lower()]
        X = df[feat_cols].fillna(0).values.astype(np.float32)
        y = df[target].fillna(0).values.astype(np.float32)
        n_in = len(feat_cols)

        if method == "shap":
            selected, scores = self._select_shap(X, y, feat_cols, entity_type)
        elif method == "rfecv":
            selected, scores = self._select_rfecv(X, y, feat_cols)
        elif method == "mutual_info":
            selected, scores = self._select_mutual_info(X, y, feat_cols)
        else:
            raise ValueError(f"Unknown selection method '{method}'. Valid: shap, rfecv, mutual_info")

        # Emit Prometheus
        try:
            from src.data.metrics import FEATURE_IMPORTANCE_UPDATE
            FEATURE_IMPORTANCE_UPDATE.labels(entity_type=entity_type, method=method).inc()
        except ImportError:
            pass

        result = FeatureSelectionResult(
            entity_type=entity_type,
            method=method,
            n_in=n_in,
            n_out=len(selected),
            selected_features=selected,
            importance_scores=scores,
            run_date=run_date.isoformat(),
        )
        result.output_path = self._write_selection(entity_type, run_date, result)
        logger.info(
            "Feature selection [%s/%s]: %d → %d features",
            entity_type, method, n_in, len(selected),
        )
        return result

    def run_full_pipeline(
        self,
        entity_type: str,
        start_date:  date,
        end_date:    date,
        target_col:  Optional[str] = None,
        method:      str = "shap",
    ) -> pd.DataFrame:
        """
        End-to-end: fetch from FeatureStoreClient → compute_features → select_features.

        Args:
            entity_type: Entity type for feature engineering and store query.
            start_date:  Window start.
            end_date:    Window end.
            target_col:  Target column for feature selection (if None, skip selection).
            method:      Feature selection method.

        Returns:
            Final feature DataFrame (selected columns only if target_col provided).
        """
        from src.data.feature_store_client import FeatureStoreClient

        fs = FeatureStoreClient()
        raw_df = fs.get_historical_features(
            start_date=start_date,
            end_date=end_date,
            entity_types=[entity_type],
        )
        engineered_df = self.compute_features(raw_df, entity_type)

        if target_col and target_col in engineered_df.columns:
            result = self.select_features(
                engineered_df, target_col, method=method,
                entity_type=entity_type, run_date=end_date,
            )
            keep = result.selected_features + [target_col]
            ts_cols = [c for c in engineered_df.columns if "timestamp" in c.lower() or c == "ds"]
            final = engineered_df[ts_cols + [c for c in keep if c in engineered_df.columns]]
        else:
            final = engineered_df

        logger.info(
            "Full pipeline [%s] %s→%s: %d rows × %d cols",
            entity_type, start_date, end_date, len(final), len(final.columns),
        )
        return final

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _select_shap(
        self, X: np.ndarray, y: np.ndarray, feat_cols: List[str], entity_type: str
    ) -> Tuple[List[str], Dict[str, float]]:
        """SHAP-based importance: train XGB on last 30d prediction subset, rank by mean |SHAP|."""
        try:
            import shap
            import xgboost as xgb
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            explainer    = shap.Explainer(model)
            shap_values  = explainer(X_val)
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            scores = {col: float(s) for col, s in zip(feat_cols, mean_abs_shap)}

            ranked   = sorted(scores, key=lambda c: scores[c], reverse=True)
            selected = ranked[:SHAP_TOP_K]
            return selected, {c: scores[c] for c in selected}

        except ImportError:
            logger.warning("SHAP or XGBoost not available; falling back to mutual_info")
            return self._select_mutual_info(X, y, feat_cols)

    def _select_rfecv(
        self, X: np.ndarray, y: np.ndarray, feat_cols: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """RFECV with XGBoost estimator, 5-fold, neg_MAE scoring."""
        try:
            import xgboost as xgb
            from sklearn.feature_selection import RFECV
            from sklearn.model_selection    import KFold

            estimator = xgb.XGBRegressor(n_estimators=50, max_depth=4, n_jobs=-1)
            selector  = RFECV(
                estimator, step=5, cv=KFold(n_splits=5, shuffle=False),
                scoring="neg_mean_absolute_error", min_features_to_select=10, n_jobs=-1,
            )
            selector.fit(X, y)
            selected = [c for c, s in zip(feat_cols, selector.support_) if s]
            # Use feature importance as proxy score
            importances = selector.estimator_.feature_importances_
            sel_mask    = selector.support_
            scores = {c: float(importances[i]) for i, c in enumerate(feat_cols) if sel_mask[i]}
            return selected, scores
        except ImportError:
            return self._select_mutual_info(X, y, feat_cols)

    @staticmethod
    def _select_mutual_info(
        X: np.ndarray, y: np.ndarray, feat_cols: List[str], threshold: float = 0.01
    ) -> Tuple[List[str], Dict[str, float]]:
        """sklearn mutual_info_regression, keep features with MI > threshold."""
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y, random_state=42)
        scores   = {col: float(v) for col, v in zip(feat_cols, mi)}
        selected = [c for c, v in scores.items() if v > threshold]
        if not selected:
            selected = sorted(scores, key=lambda c: scores[c], reverse=True)[:SHAP_TOP_K]
        return selected, {c: scores[c] for c in selected}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _write_selection(entity_type: str, run_date: date, result: FeatureSelectionResult) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"selected_features_{entity_type}_{run_date.strftime('%Y%m%d')}.json"
        path  = OUTPUT_DIR / fname
        payload = result.to_dict()
        payload["written_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Feature selection written: %s", path)
        return str(path)
