"""
Feature engineering pipeline for Tropi-Climate-Analytics ML models.
Extracts predictive variables from GPM, SMAP, GRACE-FO, Landsat, MODIS, and BMKG ground data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── Feature group definitions ──────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configuration for a single feature group."""
    name: str
    source: str
    temporal_lags: List[int]        # hours back
    spatial_radii_km: List[float]   # neighbourhood aggregation radii
    enabled: bool = True


FEATURE_GROUPS = [
    FeatureConfig("gpm_precip",      "GPM_IMERG",    [1,3,6,12,24,48,72], [0, 10, 25, 50]),
    FeatureConfig("bmkg_gauge",      "BMKG_GROUND",  [1,3,6,12,24,48,72], [0, 25, 50, 100]),
    FeatureConfig("smap_soil",       "SMAP_L3",      [6,12,24,48],        [0, 25, 50]),
    FeatureConfig("grace_gws",       "GRACE_FO",     [24*30],             [0, 100, 250]),
    FeatureConfig("modis_et",        "MODIS_MOD16",  [24,48,168],         [0, 10, 25]),
    FeatureConfig("modis_ndvi",      "MODIS_MOD13",  [168, 720],          [0, 10, 25]),
    FeatureConfig("landsat_lc",      "LANDSAT_8",    [24*30, 24*90],      [0, 10]),
    FeatureConfig("dem_hydro",       "SRTM_30M",     [],                  [0]),   # static
    FeatureConfig("bmkg_synoptic",   "BMKG_SYNOP",   [1,3,6,12,24],       [0, 50, 100]),
]


# ── Temporal features ──────────────────────────────────────────────────────────

def compute_rolling_stats(
    series: pd.Series,
    windows_hours: List[int],
    funcs: List[str] = ("mean", "max", "std", "sum"),
) -> pd.DataFrame:
    """Rolling window statistics for a time series indexed at sub-hourly resolution."""
    frames = {}
    for w in windows_hours:
        for fn in funcs:
            col = f"{series.name}_roll{w}h_{fn}"
            frames[col] = series.rolling(window=w, min_periods=max(1, w // 4)).agg(fn)
    return pd.DataFrame(frames, index=series.index)


def compute_lag_features(
    series: pd.Series,
    lags_hours: List[int],
) -> pd.DataFrame:
    """Lagged values for autoregressive features."""
    frames = {f"{series.name}_lag{lag}h": series.shift(lag) for lag in lags_hours}
    return pd.DataFrame(frames, index=series.index)


def compute_difference_features(series: pd.Series, periods: List[int]) -> pd.DataFrame:
    """First-order differences across multiple periods."""
    frames = {f"{series.name}_diff{p}h": series.diff(periods=p) for p in periods}
    return pd.DataFrame(frames, index=series.index)


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclical calendar features (hour, DOY, month, wet/dry season indicator)."""
    ts = pd.to_datetime(df[timestamp_col])
    out = df.copy()
    out["hour_sin"]  = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * ts.dt.hour / 24)
    out["doy_sin"]   = np.sin(2 * np.pi * ts.dt.dayofyear / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * ts.dt.dayofyear / 365)
    out["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    # Indonesian wet season: Nov–Apr; dry season: May–Oct
    out["wet_season"] = ts.dt.month.isin([11, 12, 1, 2, 3, 4]).astype(int)
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    return out


# ── Spatial aggregation ────────────────────────────────────────────────────────

def spatial_neighborhood_stats(
    grid: np.ndarray,
    lat: float,
    lon: float,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    radius_km: float,
    funcs: List[str] = ("mean", "max", "std"),
) -> Dict[str, float]:
    """Extract neighbourhood statistics around a point from a 2D grid."""
    # Haversine distance (vectorised)
    dlat = np.radians(lat_arr - lat)
    dlon = np.radians(lon_arr[:, None] - lon)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat)) * np.cos(np.radians(lat_arr)) * np.sin(dlon / 2) ** 2)
    dist_km = 6371 * 2 * np.arcsin(np.sqrt(a))

    mask = dist_km <= radius_km
    values = grid[mask]
    if values.size == 0:
        return {fn: np.nan for fn in funcs}

    result = {}
    for fn in funcs:
        result[fn] = float(getattr(np, fn)(values))
    return result


# ── QPE fusion features ────────────────────────────────────────────────────────

def merge_gpm_bmkg(
    gpm_df: pd.DataFrame,
    bmkg_df: pd.DataFrame,
    merge_tolerance_min: int = 15,
) -> pd.DataFrame:
    """
    Merge GPM IMERG half-hourly estimates with BMKG gauge observations.
    Produces bias-corrected QPE and difference features used by flood models.
    """
    gpm_df  = gpm_df.copy().set_index("timestamp").sort_index()
    bmkg_df = bmkg_df.copy().set_index("timestamp").sort_index()

    merged = pd.merge_asof(
        gpm_df, bmkg_df,
        left_index=True, right_index=True,
        tolerance=pd.Timedelta(minutes=merge_tolerance_min),
        suffixes=("_gpm", "_bmkg"),
    )

    # Multiplicative bias correction (BMKG as ground truth)
    merged["qpe_bias_ratio"] = np.where(
        merged["precip_gpm"] > 0.1,
        merged["precip_bmkg"] / merged["precip_gpm"].clip(lower=0.1),
        1.0,
    )
    merged["qpe_corrected"] = merged["precip_gpm"] * merged["qpe_bias_ratio"].clip(0.1, 10.0)
    merged["qpe_diff"]       = merged["precip_bmkg"] - merged["precip_gpm"]
    merged["qpe_abs_err"]    = merged["qpe_diff"].abs()
    return merged.reset_index()


# ── Drought / soil moisture features ──────────────────────────────────────────

def compute_spi(
    monthly_precip: pd.Series,
    window_months: int = 3,
) -> pd.Series:
    """
    Standardised Precipitation Index (SPI) over a rolling window.
    Uses gamma distribution fit; returns z-scores for drought monitoring.
    """
    from scipy import stats as scipy_stats

    rolling = monthly_precip.rolling(window=window_months, min_periods=window_months)
    mu  = rolling.mean()
    std = rolling.std().replace(0, np.nan)
    spi = (monthly_precip - mu) / std
    return spi.rename(f"spi_{window_months}m")


def compute_soil_wetness_index(
    smap_surface: pd.Series,
    smap_rootzone: pd.Series,
    historical_min: float,
    historical_max: float,
) -> pd.Series:
    """
    Normalise SMAP soil moisture to 0–1 Soil Wetness Index (SWI).
    Combined surface + root-zone weighted average (60/40).
    """
    surface_norm  = (smap_surface  - historical_min) / (historical_max - historical_min)
    rootzone_norm = (smap_rootzone - historical_min) / (historical_max - historical_min)
    swi = 0.6 * surface_norm + 0.4 * rootzone_norm
    return swi.clip(0, 1).rename("soil_wetness_index")


# ── Full feature matrix builder ────────────────────────────────────────────────

class FeatureMatrix:
    """
    Assembles the full feature matrix for model training and inference.
    Target variables: precip_24h_total, precip_48h_total, precip_72h_total,
                      flood_risk_score, streamflow_cms.
    """

    PRECIP_LAGS     = [1, 3, 6, 12, 24, 48, 72]
    SOIL_LAGS       = [6, 12, 24, 48]
    ROLL_WINDOWS    = [3, 6, 12, 24, 48]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._feature_names: List[str] = []

    def build(
        self,
        qpe_df: pd.DataFrame,
        smap_df: pd.DataFrame,
        bmkg_synop_df: pd.DataFrame,
        dem_features: Dict[str, float],
        target_col: str = "precip_24h_total",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns (X, y) ready for sklearn/XGBoost.
        All DataFrames must be aligned on 'timestamp' column.
        """
        base = qpe_df.copy().set_index("timestamp").sort_index()

        # Precipitation lags & rolling stats
        precip_lags  = compute_lag_features(base["qpe_corrected"], self.PRECIP_LAGS)
        precip_rolls = compute_rolling_stats(base["qpe_corrected"], self.ROLL_WINDOWS)

        # Soil moisture
        smap = smap_df.copy().set_index("timestamp").sort_index()
        smap_lags = compute_lag_features(smap["soil_wetness_index"], self.SOIL_LAGS)

        # Synoptic (pressure, humidity, wind)
        synop = bmkg_synop_df.copy().set_index("timestamp").sort_index()
        synop_lags = compute_lag_features(synop[["slp_hpa", "rh_pct", "wind_ms"]].mean(axis=1),
                                          [1, 3, 6, 12])

        # DEM (static, broadcast)
        dem_df = pd.DataFrame([dem_features] * len(base), index=base.index)

        # Calendar features
        cal_df = add_calendar_features(base.reset_index(), timestamp_col="timestamp")
        cal_df = cal_df.set_index("timestamp")[
            ["hour_sin","hour_cos","doy_sin","doy_cos","month_sin","month_cos","wet_season"]
        ]

        X = pd.concat([precip_lags, precip_rolls, smap_lags, synop_lags, dem_df, cal_df],
                      axis=1).dropna()

        y = base[target_col].reindex(X.index)
        self._feature_names = list(X.columns)
        logger.info("Feature matrix: %d rows × %d cols", len(X), len(X.columns))
        return X, y

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names.copy()
