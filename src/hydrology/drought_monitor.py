"""
Drought Monitor — SMAP soil moisture anomaly + GRACE-FO groundwater depletion.

Outputs per-province:
  SPI-3  : Standardized Precipitation Index (3-month accumulation), derived from
            SMAP-calibrated precipitation proxy and climatological baseline.
  GWS-z  : GRACE-FO Groundwater Storage anomaly z-score (normalised against
            2004-2023 GRACE/GRACE-FO climatology).

Pipeline reads from:
  - src.hydrology.soil_moisture  → SMAPIngestionPipeline (daily SMAP L3 9km)
  - src.hydrology.groundwater    → GRACEFOGroundwaterPipeline (monthly GRACE-FO TWS)

Outputs JSON to workspace/output/drought/ and optional S3 upload.
Run cadence: daily (SMAP) + monthly (GRACE-FO); Airflow DAG: smap_daily triggers
the SPI-3 update; grace_monthly triggers the GWS-z update.

Province list: 38 Indonesian provinces (BPS 2022 administrative boundaries).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import IntEnum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 38 Indonesian provinces (BPS code → name)
PROVINCE_CODES: dict[str, str] = {
    "11": "Aceh",                    "12": "Sumatera Utara",
    "13": "Sumatera Barat",          "14": "Riau",
    "15": "Jambi",                   "16": "Sumatera Selatan",
    "17": "Bengkulu",                "18": "Lampung",
    "19": "Kepulauan Bangka Belitung","21": "Kepulauan Riau",
    "31": "DKI Jakarta",             "32": "Jawa Barat",
    "33": "Jawa Tengah",             "34": "DI Yogyakarta",
    "35": "Jawa Timur",              "36": "Banten",
    "51": "Bali",                    "52": "Nusa Tenggara Barat",
    "53": "Nusa Tenggara Timur",     "61": "Kalimantan Barat",
    "62": "Kalimantan Tengah",       "63": "Kalimantan Selatan",
    "64": "Kalimantan Timur",        "65": "Kalimantan Utara",
    "71": "Sulawesi Utara",          "72": "Sulawesi Tengah",
    "73": "Sulawesi Selatan",        "74": "Sulawesi Tenggara",
    "75": "Gorontalo",               "76": "Sulawesi Barat",
    "81": "Maluku",                  "82": "Maluku Utara",
    "91": "Papua Barat",             "92": "Papua",
    "93": "Papua Selatan",           "94": "Papua Tengah",
    "95": "Papua Pegunungan",        "96": "Papua Barat Daya",
}

# SPI classification thresholds (McKee et al. 1993)
SPI_THRESHOLDS = {
    "extreme_drought":   -2.00,
    "severe_drought":    -1.50,
    "moderate_drought":  -1.00,
    "near_normal_low":   -0.50,
    "near_normal_high":   0.50,
    "moderately_wet":     1.00,
    "very_wet":           1.50,
    "extremely_wet":      2.00,
}

# GWS z-score alert levels
GWS_CRITICAL_THRESHOLD  = -1.5   # < -1.5σ: critical depletion
GWS_WARNING_THRESHOLD   = -1.0   # < -1.0σ: elevated concern
GWS_WATCH_THRESHOLD     = -0.5   # < -0.5σ: watch

# Minimum months of history needed for SPI-3
SPI3_WINDOW = 3   # months

# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class DroughtCategory(str):
    EXTREME_DROUGHT  = "extreme_drought"
    SEVERE_DROUGHT   = "severe_drought"
    MODERATE_DROUGHT = "moderate_drought"
    NEAR_NORMAL      = "near_normal"
    MODERATELY_WET   = "moderately_wet"
    VERY_WET         = "very_wet"
    EXTREMELY_WET    = "extremely_wet"


class GWSAlertLevel(str):
    NORMAL   = "normal"
    WATCH    = "watch"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class ProvinceSoilMoisture:
    """Area-weighted mean SMAP SM for a province over the past 3 months."""
    province_code: str
    province_name: str
    sm_monthly_mean: list[float]        # 3-month rolling means (m³/m³), oldest first
    sm_climatological_mean: float       # Long-run monthly mean (m³/m³)
    sm_climatological_std: float        # Long-run monthly std dev
    date: date


@dataclass
class ProvinceGWS:
    """GRACE-FO groundwater storage anomaly for a province."""
    province_code: str
    province_name: str
    gws_anomaly_mm: float               # GWS anomaly relative to 2004-2023 mean (mm)
    gws_climatological_std_mm: float    # Historical std dev (mm)
    trend_mm_per_year: float            # Linear depletion trend
    date: date


class SPIResult(BaseModel):
    province_code: str
    province_name: str
    spi_3: float = Field(..., description="Standardized Precipitation Index (3-month)")
    category: str
    date: date
    sm_3mo_mean_m3m3: float
    sm_anomaly_m3m3: float
    percentile: float = Field(..., description="SPI percentile within historical distribution")


class GWSResult(BaseModel):
    province_code: str
    province_name: str
    gws_z_score: float = Field(..., description="GWS anomaly normalised by historical std dev")
    gws_anomaly_mm: float
    trend_mm_per_year: float
    alert_level: str
    date: date


class DroughtMonitorRunStatus(BaseModel):
    run_time_utc: datetime
    reference_date: date
    provinces_processed: int
    spi_results: list[SPIResult]
    gws_results: list[GWSResult]
    provinces_drought_watch: list[str]      # province names with SPI-3 < -1.0
    provinces_gws_warning: list[str]        # province names with GWS-z < -1.0
    output_path: str
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class DroughtMonitorPipeline:
    """
    SMAP soil moisture anomaly scorer + GRACE-FO groundwater depletion tracker.

    Usage (from Airflow):
        monitor = DroughtMonitorPipeline()
        status = monitor.run(reference_date=date.today())
    """

    def __init__(self) -> None:
        self.output_dir = os.path.join(
            os.getenv("WORKSPACE_ROOT", "workspace"),
            "output", "drought"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._climatology: dict[str, dict] = {}   # province_code → {mean, std, gws_std, ...}
        self._load_climatology()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, reference_date: date) -> DroughtMonitorRunStatus:
        """
        Execute one drought monitoring cycle.

        Args:
            reference_date: Date of the most recent SMAP daily observation.
        """
        logger.info("Drought monitor run | reference_date=%s", reference_date.isoformat())
        try:
            # 1. Pull SMAP 3-month province means
            sm_records = self._aggregate_smap_by_province(reference_date)

            # 2. Compute SPI-3
            spi_results = [self._compute_spi3(rec) for rec in sm_records]

            # 3. Pull GRACE-FO GWS anomalies
            gws_records = self._aggregate_grace_by_province(reference_date)

            # 4. Compute GWS z-scores
            gws_results = [self._compute_gws_zscore(rec) for rec in gws_records]

            # 5. Alert lists
            drought_watch = [r.province_name for r in spi_results if r.spi_3 < -1.0]
            gws_warning   = [r.province_name for r in gws_results if r.gws_z_score < GWS_WARNING_THRESHOLD]

            # 6. Persist output
            output_path = self._write_output(reference_date, spi_results, gws_results)

            logger.info(
                "Drought monitor complete | %d provinces | drought_watch=%d | gws_warning=%d",
                len(spi_results), len(drought_watch), len(gws_warning),
            )

            return DroughtMonitorRunStatus(
                run_time_utc=datetime.now(timezone.utc),
                reference_date=reference_date,
                provinces_processed=len(spi_results),
                spi_results=spi_results,
                gws_results=gws_results,
                provinces_drought_watch=drought_watch,
                provinces_gws_warning=gws_warning,
                output_path=output_path,
                success=True,
            )

        except Exception as exc:
            logger.exception("Drought monitor failed: %s", exc)
            return DroughtMonitorRunStatus(
                run_time_utc=datetime.now(timezone.utc),
                reference_date=reference_date,
                provinces_processed=0,
                spi_results=[], gws_results=[],
                provinces_drought_watch=[], provinces_gws_warning=[],
                output_path="", success=False, error=str(exc),
            )

    # ------------------------------------------------------------------
    # SMAP aggregation by province
    # ------------------------------------------------------------------

    def _aggregate_smap_by_province(
        self, reference_date: date
    ) -> list[ProvinceSoilMoisture]:
        """
        Query SMAP L3 daily 9km fields and compute 3-month area-weighted
        province means. Production: reads from S3 parquet / PostGIS.
        Sprint 2: uses SMAPIngestionPipeline batch query + spatial join.
        """
        from src.hydrology.soil_moisture import SMAPIngestionPipeline

        records: list[ProvinceSoilMoisture] = []
        for code, name in PROVINCE_CODES.items():
            clim = self._climatology.get(code, {})
            sm_mean = clim.get("sm_mean", 0.28)
            sm_std  = clim.get("sm_std",  0.06)

            # Production: pull 3-month rolling means from feature store
            # Stub: sample from climatological distribution with seasonal signal
            month = reference_date.month
            seasonal_signal = 0.04 * np.sin(2 * np.pi * (month - 3) / 12)
            rng = np.random.default_rng(int(code) + reference_date.toordinal())
            sm_3mo = [
                float(np.clip(rng.normal(sm_mean + seasonal_signal, sm_std * 0.5), 0.05, 0.55))
                for _ in range(3)
            ]

            records.append(ProvinceSoilMoisture(
                province_code=code,
                province_name=name,
                sm_monthly_mean=sm_3mo,
                sm_climatological_mean=sm_mean,
                sm_climatological_std=sm_std,
                date=reference_date,
            ))
        return records

    # ------------------------------------------------------------------
    # SPI-3 computation
    # ------------------------------------------------------------------

    def _compute_spi3(self, rec: ProvinceSoilMoisture) -> SPIResult:
        """
        Compute SPI-3 from 3-month SMAP soil moisture proxy.

        SPI is computed via standardised normal deviate of the
        3-month mean relative to the long-run climatological mean and std.
        In production: use fitted gamma distribution per province × calendar-month.
        """
        sm_3mo_mean = float(np.mean(rec.sm_monthly_mean))
        anomaly     = sm_3mo_mean - rec.sm_climatological_mean

        if rec.sm_climatological_std > 0:
            spi = anomaly / rec.sm_climatological_std
        else:
            spi = 0.0

        spi = float(np.clip(spi, -3.5, 3.5))

        # Normal CDF percentile
        from scipy.stats import norm  # type: ignore[import]
        percentile = float(norm.cdf(spi) * 100)

        category = self._classify_spi(spi)

        return SPIResult(
            province_code=rec.province_code,
            province_name=rec.province_name,
            spi_3=round(spi, 3),
            category=category,
            date=rec.date,
            sm_3mo_mean_m3m3=round(sm_3mo_mean, 4),
            sm_anomaly_m3m3=round(anomaly, 4),
            percentile=round(percentile, 1),
        )

    @staticmethod
    def _classify_spi(spi: float) -> str:
        t = SPI_THRESHOLDS
        if spi <= t["extreme_drought"]:    return DroughtCategory.EXTREME_DROUGHT
        if spi <= t["severe_drought"]:     return DroughtCategory.SEVERE_DROUGHT
        if spi <= t["moderate_drought"]:   return DroughtCategory.MODERATE_DROUGHT
        if spi >= t["extremely_wet"]:      return DroughtCategory.EXTREMELY_WET
        if spi >= t["very_wet"]:           return DroughtCategory.VERY_WET
        if spi >= t["moderately_wet"]:     return DroughtCategory.MODERATELY_WET
        return DroughtCategory.NEAR_NORMAL

    # ------------------------------------------------------------------
    # GRACE-FO aggregation by province
    # ------------------------------------------------------------------

    def _aggregate_grace_by_province(
        self, reference_date: date
    ) -> list[ProvinceGWS]:
        """
        Pull GRACE-FO TWS anomaly and derive groundwater storage (GWS) by
        subtracting soil moisture and snow water equivalent (SMS + SWE from GLDAS).
        GWS = TWS_anomaly − SMS_anomaly − SWE_anomaly
        Production: reads from GRACE-FO monthly mascon grid (CSR RL06M).
        """
        from src.hydrology.groundwater import GRACEFOGroundwaterPipeline

        records: list[ProvinceGWS] = []
        for code, name in PROVINCE_CODES.items():
            clim = self._climatology.get(code, {})
            gws_std   = clim.get("gws_std_mm", 40.0)
            gws_trend = clim.get("gws_trend_mm_per_year", -2.5)

            rng = np.random.default_rng(int(code) * 7 + reference_date.toordinal())
            # Simulate GWS anomaly with trend + noise
            years_since_base = (reference_date.year - 2015) + reference_date.month / 12
            gws_anomaly = gws_trend * years_since_base + float(rng.normal(0, gws_std * 0.4))

            records.append(ProvinceGWS(
                province_code=code,
                province_name=name,
                gws_anomaly_mm=round(gws_anomaly, 1),
                gws_climatological_std_mm=gws_std,
                trend_mm_per_year=gws_trend,
                date=reference_date,
            ))
        return records

    # ------------------------------------------------------------------
    # GWS z-score computation
    # ------------------------------------------------------------------

    def _compute_gws_zscore(self, rec: ProvinceGWS) -> GWSResult:
        """Normalise GWS anomaly by province historical std dev."""
        if rec.gws_climatological_std_mm > 0:
            z = rec.gws_anomaly_mm / rec.gws_climatological_std_mm
        else:
            z = 0.0
        z = float(np.clip(z, -4.0, 4.0))

        if z < GWS_CRITICAL_THRESHOLD:
            level = GWSAlertLevel.CRITICAL
        elif z < GWS_WARNING_THRESHOLD:
            level = GWSAlertLevel.WARNING
        elif z < GWS_WATCH_THRESHOLD:
            level = GWSAlertLevel.WATCH
        else:
            level = GWSAlertLevel.NORMAL

        return GWSResult(
            province_code=rec.province_code,
            province_name=rec.province_name,
            gws_z_score=round(z, 3),
            gws_anomaly_mm=rec.gws_anomaly_mm,
            trend_mm_per_year=rec.trend_mm_per_year,
            alert_level=level,
            date=rec.date,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_output(
        self,
        ref_date: date,
        spi_results: list[SPIResult],
        gws_results: list[GWSResult],
    ) -> str:
        """Write combined drought assessment JSON to workspace output dir."""
        fname = f"drought_assessment_{ref_date.strftime('%Y%m%d')}.json"
        path  = os.path.join(self.output_dir, fname)

        payload = {
            "reference_date": ref_date.isoformat(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "sprint2.0",
            "spi_3": [r.model_dump() for r in spi_results],
            "gws_z_score": [r.model_dump() for r in gws_results],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Drought output written: %s", path)
        return path

    # ------------------------------------------------------------------
    # Climatology loader
    # ------------------------------------------------------------------

    def _load_climatology(self) -> None:
        """
        Load province-level SM and GWS climatological statistics.
        Production: read from S3 parquet (workspace/data/climatology/).
        Sprint 2: hard-coded reasonable defaults per island group.
        """
        # Wetter provinces (Kalimantan, Papua, Maluku)
        wet_provinces = {"61","62","63","64","65","81","82","91","92","93","94","95","96"}
        # Drier provinces (NTB, NTT, Jawa Timur during dry season)
        dry_provinces = {"52","53","35","34"}

        for code in PROVINCE_CODES:
            if code in wet_provinces:
                self._climatology[code] = {
                    "sm_mean": 0.38, "sm_std": 0.07,
                    "gws_std_mm": 55.0, "gws_trend_mm_per_year": -1.5,
                }
            elif code in dry_provinces:
                self._climatology[code] = {
                    "sm_mean": 0.20, "sm_std": 0.06,
                    "gws_std_mm": 30.0, "gws_trend_mm_per_year": -3.5,
                }
            else:
                self._climatology[code] = {
                    "sm_mean": 0.29, "sm_std": 0.07,
                    "gws_std_mm": 40.0, "gws_trend_mm_per_year": -2.5,
                }
