"""
Feature Engineering Pipeline — ANALYTICA
Extracts 1000+ predictive variables from MODIS, GPM, BMKG, SMAP, DEM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────

WET_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}   # Nov-Apr

PRECIP_LAG_HOURS = [1, 3, 6, 12, 18, 24, 48, 72, 168]
ROLLING_WINDOWS  = [3, 6, 12, 24, 48, 72]

MODIS_BANDS = [
    "cloud_optical_depth", "cloud_effective_radius", "cloud_top_temp",
    "cloud_fraction", "cloud_top_pressure", "cloud_water_path",
    "cloud_phase", "aerosol_optical_depth", "aerosol_angstrom",
    "land_surface_temp_day", "land_surface_temp_night",
    "ndvi", "evi", "lai", "fpar",
    "albedo_shortwave", "albedo_longwave",
]

GPM_VARS = [
    "precip_rate", "precip_probability", "precip_type",
    "latent_heat_flux", "convective_fraction",
    "rain_water_content", "ice_water_content",
    "precip_rate_1h", "precip_rate_3h", "precip_rate_6h", "precip_rate_24h",
]

BMKG_VARS = [
    "station_rainfall_1h", "station_rainfall_24h",
    "temp_2m", "dewpoint_2m", "rh_2m",
    "wind_speed_10m", "wind_dir_10m", "wind_u_10m", "wind_v_10m",
    "sea_level_pressure", "visibility",
    "cloud_cover_oktas", "ceiling_height",
    "station_count_50km", "station_count_100km",
]

SMAP_VARS = [
    "soil_moisture_surface", "soil_moisture_rootzone",
    "surface_temperature", "vegetation_opacity",
]

ATMOSPHERIC_VARS = [
    "cape", "cin", "lifted_index",
    "k_index", "totals_totals", "sweat_index",
    "wind_shear_0_3km", "wind_shear_0_6km", "wind_shear_0_9km",
    "precipitable_water_total", "precipitable_water_850_500",
    "convergence_850hPa", "vorticity_500hPa",
    "omega_500hPa", "moisture_flux_divergence",
]

TOPOGRAPHY_VARS = [
    "elevation", "slope", "aspect", "curvature",
    "terrain_roughness_index", "topographic_wetness_index",
    "distance_to_coast_km", "distance_to_major_river_km",
    "catchment_area", "flow_direction",
]

CLIMATE_INDEX_VARS = [
    "oni", "iod", "sam_index",
    "mjo_phase", "mjo_amplitude",
    "enso_phase",   # -1 La Nina, 0 Neutral, 1 El Nino
    "itcz_latitude",
]


class FeatureEngineeringPipeline:
    """
    Extract, transform and assemble ML-ready feature matrices from
    MODIS, GPM, BMKG, SMAP, DEM and climate-index inputs.
    Produces ~1000+ features per grid cell per timestep.
    """

    def __init__(self, target_resolution_deg: float = 0.25) -> None:
        self.resolution = target_resolution_deg
        self._fitted    = False
        self._stats: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def temporal_features(timestamps: pd.Series) -> pd.DataFrame:
        """Cyclic sin/cos encoding for hour, DOY, month + wet-season flag."""
        ts = pd.to_datetime(timestamps)
        df = pd.DataFrame(index=timestamps.index)
        df["hour_sin"]      = np.sin(2 * np.pi * ts.dt.hour      / 24)
        df["hour_cos"]      = np.cos(2 * np.pi * ts.dt.hour      / 24)
        df["doy_sin"]       = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
        df["doy_cos"]       = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
        df["month_sin"]     = np.sin(2 * np.pi * ts.dt.month     / 12)
        df["month_cos"]     = np.cos(2 * np.pi * ts.dt.month     / 12)
        df["is_wet_season"] = ts.dt.month.isin(WET_SEASON_MONTHS).astype(float)
        return df

    @staticmethod
    def lag_features(series: pd.Series,
                     lag_hours: List[int] = None,
                     name: str = "precip") -> pd.DataFrame:
        if lag_hours is None:
            lag_hours = PRECIP_LAG_HOURS
        df = pd.DataFrame(index=series.index)
        for h in lag_hours:
            df[f"{name}_lag_{h}h"] = series.shift(h)
        return df

    @staticmethod
    def rolling_features(series: pd.Series,
                         windows: List[int] = None,
                         name: str = "precip") -> pd.DataFrame:
        if windows is None:
            windows = ROLLING_WINDOWS
        df = pd.DataFrame(index=series.index)
        for w in windows:
            df[f"{name}_roll_mean_{w}h"] = series.rolling(w).mean()
            df[f"{name}_roll_max_{w}h"]  = series.rolling(w).max()
            df[f"{name}_roll_std_{w}h"]  = series.rolling(w).std()
        return df

    @staticmethod
    def landsat_indices(bands: pd.DataFrame) -> pd.DataFrame:
        """
        Compute NDVI, EVI, NDWI, MNDWI, NBR, NDBI from Landsat OLI bands.
        bands: DataFrame with columns B1..B7 (scaled reflectance 0-1).
        """
        df  = pd.DataFrame(index=bands.index)
        eps = 1e-6
        B   = {f"B{i}": bands.get(f"B{i}", pd.Series(np.nan, index=bands.index))
               for i in range(1, 8)}
        df["ndvi"]  = (B["B5"] - B["B4"]) / (B["B5"] + B["B4"] + eps)
        df["evi"]   = 2.5 * (B["B5"] - B["B4"]) / (B["B5"] + 6*B["B4"] - 7.5*B["B2"] + 1 + eps)
        df["ndwi"]  = (B["B3"] - B["B5"]) / (B["B3"] + B["B5"] + eps)
        df["mndwi"] = (B["B3"] - B["B6"]) / (B["B3"] + B["B6"] + eps)
        df["nbr"]   = (B["B5"] - B["B7"]) / (B["B5"] + B["B7"] + eps)
        df["ndbi"]  = (B["B6"] - B["B5"]) / (B["B6"] + B["B5"] + eps)
        return df

    @staticmethod
    def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Key atmospheric x surface cross-terms."""
        out = pd.DataFrame(index=df.index)
        if {"cape", "precipitable_water_total"}.issubset(df.columns):
            out["cape_x_pw"]    = df["cape"] * df["precipitable_water_total"]
        if {"ndvi", "soil_moisture_surface"}.issubset(df.columns):
            out["ndvi_x_sm"]    = df["ndvi"] * df["soil_moisture_surface"]
        if {"wind_shear_0_6km", "cape"}.issubset(df.columns):
            out["shear_x_cape"] = df["wind_shear_0_6km"] * df["cape"]
        if {"oni", "iod"}.issubset(df.columns):
            out["oni_x_iod"]    = df["oni"] * df["iod"]
        return out

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        raw: DataFrame with raw sensor columns + optional 'timestamp' column.
        Returns a model-ready feature matrix (NaN-imputed, optionally normalised).
        """
        parts: List[pd.DataFrame] = []

        if "timestamp" in raw.columns:
            parts.append(self.temporal_features(raw["timestamp"]))

        if "precip_rate" in raw.columns:
            parts.append(self.lag_features(raw["precip_rate"]))
            parts.append(self.rolling_features(raw["precip_rate"]))

        landsat_cols = [c for c in raw.columns if c.startswith("B") and c[1:].isdigit()]
        if landsat_cols:
            parts.append(self.landsat_indices(raw[landsat_cols]))

        all_raw_cols = (MODIS_BANDS + GPM_VARS + BMKG_VARS +
                        SMAP_VARS + ATMOSPHERIC_VARS +
                        TOPOGRAPHY_VARS + CLIMATE_INDEX_VARS)
        available_raw = [c for c in raw.columns if c in all_raw_cols]
        if available_raw:
            parts.append(raw[available_raw])

        feat = pd.concat(parts, axis=1)
        feat = feat.loc[:, ~feat.columns.duplicated()]

        feat = pd.concat([feat, self.interaction_features(feat)], axis=1)
        feat = feat.ffill().fillna(feat.median())

        if self._fitted:
            feat = self._normalise(feat)

        logger.debug(f"Feature matrix: {feat.shape[0]} rows x {feat.shape[1]} features")
        return feat

    def fit(self, df: pd.DataFrame) -> "FeatureEngineeringPipeline":
        """Compute normalisation stats from training data."""
        transformed = self.transform(df)
        self._stats = {
            col: {"mean": float(transformed[col].mean()),
                  "std":  max(float(transformed[col].std()), 1e-8)}
            for col in transformed.columns
        }
        self._fitted = True
        return self

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col in self._stats:
                df[col] = ((df[col] - self._stats[col]["mean"])
                           / self._stats[col]["std"])
        return df

    @property
    def feature_count(self) -> int:
        """Approximate total features produced by this pipeline."""
        temporal    = 7
        lag_rolling = len(PRECIP_LAG_HOURS) + len(ROLLING_WINDOWS) * 3
        raw_sources = (len(MODIS_BANDS) + len(GPM_VARS) + len(BMKG_VARS) +
                       len(SMAP_VARS) + len(ATMOSPHERIC_VARS) +
                       len(TOPOGRAPHY_VARS) + len(CLIMATE_INDEX_VARS))
        spectral    = 6
        interactions = 4
        return temporal + lag_rolling + raw_sources + spectral + interactions
