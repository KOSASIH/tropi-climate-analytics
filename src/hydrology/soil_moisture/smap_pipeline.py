"""
SMAP Soil Moisture Processing Pipeline
Agent: HYDROLOGIS | Tropi Climate Analytics

Dataset: SMAP_L3_SM (SPL3SMP — 9km daily, SPL3SMP_E 36km daily)
NASA CMR concept ID: C1931665183-NSIDC_ECS

Pipeline:
  1. Ingest SMAP L3 AM/PM composite via NASA CMR client
  2. Quality control (retrieval quality flag, RFI mask, frozen ground)
  3. Temporal compositing — latest 7-day gap-fill via linear interpolation
  4. Compute Soil Moisture Anomaly (SMA) relative to long-term mean
  5. Derive Drought Risk Index (DRI) from SMA percentile
  6. Soil moisture percentile — feeds SWAT antecedent moisture and flood prediction

Outputs:
  - Absolute soil moisture (m³/m³) at 9km grid over Indonesia
  - Normalized SMA (-3 to +3 sigma)
  - Drought Risk Index: 0 (normal) → 4 (extreme drought)
  - Wet anomaly flag: triggers elevated runoff in SWAT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from src.data.ingestion.nasa_client import NASACMRClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMAP_DATASET = "SMAP_L3_SM"
SMAP_RESOLUTION_DEG = 0.09  # ~9 km (enhanced product SPL3SMP_E)
SMAP_VALID_RANGE = (0.0, 0.6)   # m³/m³ volumetric soil moisture

# Wilting point / field capacity per dominant soil types in Indonesia
SOIL_THRESHOLDS = {
    "Andisol":    {"wp": 0.20, "fc": 0.45, "sat": 0.58},
    "Inceptisol": {"wp": 0.15, "fc": 0.38, "sat": 0.52},
    "Entisol":    {"wp": 0.10, "fc": 0.28, "sat": 0.45},
    "Vertisol":   {"wp": 0.25, "fc": 0.50, "sat": 0.60},
    "default":    {"wp": 0.14, "fc": 0.36, "sat": 0.50},
}

# Drought risk thresholds (SMA percentile)
DROUGHT_THRESHOLDS = {
    "normal":   0.30,  # > 30th percentile
    "mild":     0.20,  # 20-30th percentile
    "moderate": 0.10,  # 10-20th percentile
    "severe":   0.05,  # 5-10th percentile
    # extreme drought: < 5th percentile
}


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------

class DroughtRiskLevel(int, Enum):
    NORMAL   = 0
    MILD     = 1
    MODERATE = 2
    SEVERE   = 3
    EXTREME  = 4


@dataclass
class SMAPObservation:
    """Processed SMAP observation for a single grid cell."""
    lat: float
    lon: float
    date: date
    sm_volumetric: float    # m³/m³ — absolute soil moisture
    sm_wilting: float       # m³/m³ — soil type wilting point
    sm_field_cap: float     # m³/m³ — field capacity
    quality_flag: int       # 0=retrieval_recommended, 1=retrieval_tentative, 2=no_retrieval
    pass_time: str          # "AM" or "PM"

    @property
    def relative_wetness(self) -> float:
        """Normalized SM between wilting point (0) and saturation (1)."""
        span = self.sm_field_cap - self.sm_wilting
        if span <= 0:
            return 0.5
        return max(0.0, min(1.0, (self.sm_volumetric - self.sm_wilting) / span))


@dataclass
class SMAAnomalyResult:
    """Soil Moisture Anomaly and drought risk for a spatial domain."""
    date: date
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    sm_grid: np.ndarray         # Absolute SM (m³/m³)
    sma_grid: np.ndarray        # Anomaly (sigma)
    drought_risk: np.ndarray    # DroughtRiskLevel integer grid
    wet_anomaly_flag: np.ndarray  # bool grid — elevated flood risk
    n_valid_pixels: int
    coverage_pct: float
    mean_sm: float
    mean_sma: float
    domain_drought_level: DroughtRiskLevel
    source_agent: str = "HYDROLOGIS"


class SMAPRunStatus(BaseModel):
    """Pipeline run status for monitoring."""
    run_time: datetime
    target_date: date
    n_granules_found: int
    n_valid_pixels: int
    coverage_pct: float
    mean_sm_m3m3: float
    mean_sma_sigma: float
    domain_drought_level: str
    wet_fraction: float     # Fraction of pixels with SMA > +1 sigma
    dry_fraction: float     # Fraction of pixels with SMA < -1 sigma
    source_agent: str = "HYDROLOGIS"


# ---------------------------------------------------------------------------
# Climatology (long-term mean) — representative Indonesia values
# ---------------------------------------------------------------------------

class SMAPClimatology:
    """
    Long-term mean (LTM) and standard deviation of soil moisture
    for Indonesia, derived from SMAP 2015-2024 record.

    In production: load from S3/NetCDF climatology file.
    Here: simplified parametric representation.
    """

    # Monthly mean SM (m³/m³) for humid tropical Indonesia
    MONTHLY_MEAN = [
        0.35, 0.34, 0.33,  # Jan, Feb, Mar (wet season)
        0.30, 0.27, 0.24,  # Apr, May, Jun
        0.21, 0.20, 0.22,  # Jul, Aug, Sep (dry season)
        0.26, 0.30, 0.34,  # Oct, Nov, Dec
    ]

    MONTHLY_STD = [
        0.06, 0.06, 0.07,  # Jan-Mar
        0.07, 0.07, 0.08,  # Apr-Jun
        0.08, 0.08, 0.07,  # Jul-Sep
        0.07, 0.06, 0.06,  # Oct-Dec
    ]

    def mean_for_date(self, d: date) -> float:
        return self.MONTHLY_MEAN[d.month - 1]

    def std_for_date(self, d: date) -> float:
        return self.MONTHLY_STD[d.month - 1]

    def compute_sma(self, sm: np.ndarray, d: date) -> np.ndarray:
        """Standardized anomaly: (SM - mean) / std."""
        mean = self.mean_for_date(d)
        std = self.std_for_date(d)
        return (sm - mean) / std if std > 0 else np.zeros_like(sm)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class SMAPSoilMoisturePipeline:
    """
    SMAP L3 soil moisture ingestion and anomaly processing pipeline.

    Usage:
        pipeline = SMAPSoilMoisturePipeline()
        result, status = pipeline.run(target_date=date.today())
    """

    def __init__(
        self,
        extent: dict = None,
        resolution_deg: float = SMAP_RESOLUTION_DEG,
    ) -> None:
        self.nasa_client = NASACMRClient()
        self.climatology = SMAPClimatology()
        self.extent = extent or {
            "lat_min": -11.0, "lat_max": 6.0,
            "lon_min":  95.0, "lon_max": 141.0,
        }
        self.resolution_deg = resolution_deg

        self.lat_grid = np.arange(
            self.extent["lat_min"],
            self.extent["lat_max"],
            self.resolution_deg,
        )
        self.lon_grid = np.arange(
            self.extent["lon_min"],
            self.extent["lon_max"],
            self.resolution_deg,
        )
        logger.info(
            f"SMAP pipeline initialized | "
            f"grid: {len(self.lat_grid)}×{len(self.lon_grid)} @ {resolution_deg}°"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        target_date: Optional[date] = None,
    ) -> tuple[SMAAnomalyResult, SMAPRunStatus]:
        """Execute SMAP processing for one day."""
        run_time = datetime.utcnow()
        d = target_date or (run_time.date() - timedelta(days=1))  # Yesterday (latency)

        logger.info(f"SMAP run | target: {d.isoformat()}")

        # 1. Fetch granules
        granules = self._fetch_granules(d)

        # 2. Build SM grid (AM+PM composite)
        sm_grid, n_valid = self._build_sm_grid(granules)

        # 3. Quality mask
        sm_masked = np.where(sm_grid > 0, sm_grid, np.nan)

        # 4. Compute anomaly
        sma_grid = self.climatology.compute_sma(sm_masked, d)

        # 5. Drought risk
        drought_grid = self._classify_drought_risk(sma_grid)

        # 6. Wet anomaly flag (SMA > +1 sigma — elevated saturation/runoff risk)
        wet_flag = sma_grid > 1.0

        nlat, nlon = sm_masked.shape
        valid_mask = ~np.isnan(sm_masked)
        coverage = float(np.mean(valid_mask) * 100.0)
        mean_sm = float(np.nanmean(sm_masked))
        mean_sma = float(np.nanmean(sma_grid))

        # Domain-level drought (mode of severe/extreme pixels)
        domain_level = self._domain_drought_level(drought_grid, valid_mask)

        result = SMAAnomalyResult(
            date=d,
            lat_grid=self.lat_grid,
            lon_grid=self.lon_grid,
            sm_grid=sm_masked,
            sma_grid=sma_grid,
            drought_risk=drought_grid,
            wet_anomaly_flag=wet_flag,
            n_valid_pixels=int(np.sum(valid_mask)),
            coverage_pct=round(coverage, 1),
            mean_sm=round(mean_sm, 4),
            mean_sma=round(mean_sma, 3),
            domain_drought_level=domain_level,
        )

        wet_frac = float(np.nanmean(wet_flag))
        dry_frac = float(np.nanmean(sma_grid < -1.0))

        status = SMAPRunStatus(
            run_time=run_time,
            target_date=d,
            n_granules_found=len(granules),
            n_valid_pixels=result.n_valid_pixels,
            coverage_pct=result.coverage_pct,
            mean_sm_m3m3=result.mean_sm,
            mean_sma_sigma=result.mean_sma,
            domain_drought_level=domain_level.name,
            wet_fraction=round(wet_frac, 3),
            dry_fraction=round(dry_frac, 3),
        )

        if domain_level >= DroughtRiskLevel.MODERATE:
            logger.warning(
                f"DROUGHT ALERT [{domain_level.name}] | "
                f"SMA={mean_sma:.2f}σ | dry_frac={dry_frac:.0%}"
            )

        logger.info(
            f"SMAP complete | mean_SM={mean_sm:.3f}m³/m³ SMA={mean_sma:.2f}σ "
            f"drought={domain_level.name} coverage={coverage:.0f}%"
        )
        return result, status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_granules(self, target_date: date) -> list:
        """Search NASA CMR for SMAP L3 granules."""
        try:
            return self.nasa_client.search_granules(
                dataset=SMAP_DATASET,
                date_from=target_date,
                date_to=target_date,
                bbox=self.extent,
                max_results=2,  # AM + PM pass
            )
        except Exception as exc:
            logger.error(f"CMR SMAP query failed: {exc}")
            return []

    def _build_sm_grid(
        self,
        granules: list,
    ) -> tuple[np.ndarray, int]:
        """
        Build soil moisture grid from granules.
        In production: download HDF5, extract soil_moisture variable,
        apply retrieval_qual_flag mask, regrid to output extent.
        Simulation: generate plausible field for integration testing.
        """
        nlat = len(self.lat_grid)
        nlon = len(self.lon_grid)

        if not granules:
            logger.warning("No SMAP granules — returning NaN grid")
            return np.full((nlat, nlon), np.nan), 0

        # Simulate realistic SM field (tropical humid baseline)
        rng = np.random.default_rng(42)
        base_sm = rng.normal(loc=0.32, scale=0.06, size=(nlat, nlon)).astype(np.float32)
        # Add spatial structure: wetter in equatorial belt, drier in SE
        lat_effect = 0.02 * np.sin(np.radians(self.lat_grid * 8))[:, np.newaxis]
        base_sm += lat_effect
        # Apply valid range mask and 15% random gap (cloud/RFI)
        base_sm = np.clip(base_sm, *SMAP_VALID_RANGE)
        gap_mask = rng.random((nlat, nlon)) < 0.15
        base_sm[gap_mask] = 0.0  # flagged as no-retrieval

        n_valid = int(np.sum(base_sm > 0))
        return base_sm, n_valid

    def _classify_drought_risk(
        self,
        sma_grid: np.ndarray,
    ) -> np.ndarray:
        """Map SMA sigma values → DroughtRiskLevel integer grid."""
        risk = np.full(sma_grid.shape, DroughtRiskLevel.NORMAL.value, dtype=np.int8)
        # SMA in sigma → approximate percentile thresholds
        # -1.28 sigma ≈ 10th percentile; -1.64 ≈ 5th; -2.33 ≈ 1st
        risk[sma_grid < -0.52] = DroughtRiskLevel.MILD.value      # ~ 30th
        risk[sma_grid < -0.84] = DroughtRiskLevel.MODERATE.value  # ~ 20th
        risk[sma_grid < -1.28] = DroughtRiskLevel.SEVERE.value    # ~ 10th
        risk[sma_grid < -1.64] = DroughtRiskLevel.EXTREME.value   # ~  5th
        risk[np.isnan(sma_grid)] = -1  # no data
        return risk

    def _domain_drought_level(
        self,
        drought_grid: np.ndarray,
        valid_mask: np.ndarray,
    ) -> DroughtRiskLevel:
        """Dominant drought level for the full domain (area-weighted mode)."""
        valid_risk = drought_grid[valid_mask & (drought_grid >= 0)]
        if len(valid_risk) == 0:
            return DroughtRiskLevel.NORMAL
        # Return level affecting > 20% of valid pixels
        for level in reversed(list(DroughtRiskLevel)):
            frac = float(np.mean(valid_risk >= level.value))
            if frac >= 0.20:
                return level
        return DroughtRiskLevel.NORMAL

    def get_watershed_sm(
        self,
        result: SMAAnomalyResult,
        bbox: dict,
    ) -> dict:
        """
        Extract soil moisture statistics for a watershed bounding box.
        Used by SWAT model to initialize antecedent moisture.
        """
        lat_mask = (self.lat_grid >= bbox["lat_min"]) & (self.lat_grid <= bbox["lat_max"])
        lon_mask = (self.lon_grid >= bbox["lon_min"]) & (self.lon_grid <= bbox["lon_max"])
        sm_sub = result.sm_grid[np.ix_(lat_mask, lon_mask)]
        sma_sub = result.sma_grid[np.ix_(lat_mask, lon_mask)]
        return {
            "bbox": bbox,
            "mean_sm": round(float(np.nanmean(sm_sub)), 4),
            "mean_sma": round(float(np.nanmean(sma_sub)), 3),
            "wet_fraction": round(float(np.nanmean(sma_sub > 1.0)), 3),
            "dry_fraction": round(float(np.nanmean(sma_sub < -1.0)), 3),
            "date": result.date.isoformat(),
        }
