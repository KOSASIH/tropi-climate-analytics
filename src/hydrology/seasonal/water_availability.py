"""
Seasonal Water Availability Forecasts for Agricultural Planning
Agent: HYDROLOGIS | Tropi Climate Analytics

Outputs seasonal water outlook (3-month rolling) across Indonesian
agricultural regions, integrating:
  - SMAP soil moisture trend
  - GRACE-FO groundwater storage
  - QPE precipitation accumulation
  - SWAT baseflow and reservoir inflow simulation
  - ENSO state (El Niño / La Niña) for inter-annual modulation

Products:
  1. Water Availability Index (WAI) per river basin / province: 0-100 scale
  2. Irrigation water deficit/surplus (m³/ha/season)
  3. Reservoir inflow forecast (MCM/month)
  4. Paddy rice planting window recommendation
  5. Drought early warning for rain-fed agriculture

Agricultural calendar alignment (Indonesian monsoon):
  Planting Season 1 (MT-I):  Oct-Mar (wet season)
  Planting Season 2 (MT-II): Apr-Sep (dry season, irrigation-dependent)
  MT-II is highest risk for drought — primary planning window

Target users: Ministry of Agriculture (Kementan), BAPPENAS, Badan Ketahanan Pangan
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indonesian planting season windows
PLANTING_SEASONS = {
    "MT-I":  {"months": [10, 11, 12, 1, 2, 3],  "label": "Wet Season (Oct-Mar)"},
    "MT-II": {"months": [4, 5, 6, 7, 8, 9],     "label": "Dry Season (Apr-Sep)"},
}

# Target agricultural basins
AGRI_BASINS = {
    "brantas_delta": {
        "name": "Brantas Delta — East Java",
        "province": "Jawa Timur",
        "lat": -7.4, "lon": 112.7,
        "irrigated_area_ha": 210_000,
        "rain_fed_area_ha": 180_000,
        "reservoir": "Sutami-Lahor",
        "reservoir_capacity_mcm": 343.0,
        "crop": "paddy_rice",
    },
    "solo_bengawan": {
        "name": "Bengawan Solo Basin — Central Java",
        "province": "Jawa Tengah",
        "lat": -7.2, "lon": 111.5,
        "irrigated_area_ha": 340_000,
        "rain_fed_area_ha": 220_000,
        "reservoir": "Wonogiri",
        "reservoir_capacity_mcm": 730.0,
        "crop": "paddy_rice",
    },
    "citarum": {
        "name": "Citarum Basin — West Java",
        "province": "Jawa Barat",
        "lat": -6.9, "lon": 107.4,
        "irrigated_area_ha": 280_000,
        "rain_fed_area_ha": 150_000,
        "reservoir": "Jatiluhur (Ir. H. Djuanda)",
        "reservoir_capacity_mcm": 2_960.0,
        "crop": "paddy_rice",
    },
    "pompengan_jeneberang": {
        "name": "Pompengan-Jeneberang — South Sulawesi",
        "province": "Sulawesi Selatan",
        "lat": -5.1, "lon": 119.5,
        "irrigated_area_ha": 89_000,
        "rain_fed_area_ha": 120_000,
        "reservoir": "Bilibili",
        "reservoir_capacity_mcm": 375.0,
        "crop": "paddy_rice",
    },
}

# Paddy rice water requirement (mm/day)
PADDY_WATER_REQ_MM_DAY = {
    "land_prep":     12.0,
    "tillering":      8.0,
    "heading":        9.0,
    "grain_filling":  7.0,
    "ripening":       4.0,
}

# ENSO modulation factors for Indonesian precipitation
ENSO_PRECIP_FACTOR = {
    "strong_la_nina": 1.30,   # +30% above normal
    "weak_la_nina":   1.12,
    "neutral":        1.00,
    "weak_el_nino":   0.82,
    "strong_el_nino": 0.65,   # -35% below normal
}


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------

class WaterAvailabilityClass(str, Enum):
    ABUNDANT   = "abundant"    # WAI 80-100: surplus > 20%
    SUFFICIENT = "sufficient"  # WAI 60-80: near-normal
    MODERATE   = "moderate"    # WAI 40-60: mild deficit
    SCARCE     = "scarce"      # WAI 20-40: significant deficit
    CRITICAL   = "critical"    # WAI 0-20:  crisis


class PlantingRecommendation(str, Enum):
    PROCEED_ON_SCHEDULE    = "proceed_on_schedule"
    PROCEED_WITH_CAUTION   = "proceed_with_caution"
    DELAY_2_WEEKS          = "delay_2_weeks"
    DELAY_4_WEEKS          = "delay_4_weeks"
    SWITCH_TO_PALAWIJA     = "switch_to_palawija"  # Short-season crop
    SUSPEND_RAIN_FED       = "suspend_rain_fed"


class ENSOState(str, Enum):
    STRONG_LA_NINA = "strong_la_nina"
    WEAK_LA_NINA   = "weak_la_nina"
    NEUTRAL        = "neutral"
    WEAK_EL_NINO   = "weak_el_nino"
    STRONG_EL_NINO = "strong_el_nino"


@dataclass
class SeasonalOutlook:
    """3-month water availability outlook for one basin."""
    basin_id: str
    basin_name: str
    province: str
    issued_date: date
    season: str                         # MT-I or MT-II
    forecast_months: list[str]          # e.g. ["2026-06", "2026-07", "2026-08"]
    wai_score: float                    # Water Availability Index 0-100
    availability_class: WaterAvailabilityClass
    precip_forecast_mm: list[float]     # Monthly total (mm)
    precip_normal_mm: list[float]       # Climatological normal
    precip_anomaly_pct: list[float]     # % departure from normal
    soil_moisture_m3m3: float           # Current SMAP reading
    groundwater_anomaly_cm: float       # GRACE-FO GWS anomaly
    reservoir_storage_pct: float        # Current reservoir fill %
    irrigation_deficit_m3ha: float      # Irrigation water deficit (m³/ha)
    planting_recommendation: PlantingRecommendation
    enso_state: ENSOState
    confidence_pct: float
    advisory: str                       # Human-readable advisory message


class SeasonalForecastStatus(BaseModel):
    """Pipeline run metadata."""
    run_time: datetime
    issued_date: date
    n_basins: int
    n_critical_basins: int
    n_scarce_basins: int
    most_critical_basin: Optional[str]
    enso_state: str
    source_agent: str = "HYDROLOGIS"


# ---------------------------------------------------------------------------
# ENSO state reader
# ---------------------------------------------------------------------------

class ENSOMonitor:
    """
    Reads ENSO state from NOAA Niño-3.4 SST anomaly index.
    In production: subscribe to NOAA CPC monthly ENSO update.
    """

    # Climatological monthly precipitation (mm) per basin
    MONTHLY_NORMAL = {
        "brantas_delta":         [230, 210, 195, 140, 90, 55, 35, 30, 55, 110, 175, 215],
        "solo_bengawan":         [290, 265, 250, 175, 95, 50, 30, 25, 65, 130, 200, 270],
        "citarum":               [300, 280, 260, 195, 120, 65, 40, 38, 80, 150, 220, 295],
        "pompengan_jeneberang":  [200, 175, 160, 145, 120, 110, 85, 70, 90, 130, 165, 195],
    }

    def get_current_state(self) -> ENSOState:
        """Return current ENSO state (live query in production)."""
        # Simulation: return weak El Niño for June 2026
        return ENSOState.WEAK_EL_NINO

    def forecast_precip(
        self,
        basin_id: str,
        target_months: list[int],  # month numbers [1-12]
        enso_state: ENSOState,
    ) -> tuple[list[float], list[float]]:
        """
        Returns (forecast_mm, normal_mm) for each target month.
        """
        normals = self.MONTHLY_NORMAL.get(
            basin_id, self.MONTHLY_NORMAL["brantas_delta"]
        )
        factor = ENSO_PRECIP_FACTOR[enso_state.value]
        forecast = [normals[m - 1] * factor for m in target_months]
        normal = [normals[m - 1] for m in target_months]
        return forecast, normal


# ---------------------------------------------------------------------------
# Water Availability Index calculator
# ---------------------------------------------------------------------------

class WAICalculator:
    """
    Composite Water Availability Index (WAI) 0-100.

    Components:
      - Precipitation anomaly (40% weight)
      - Soil moisture percentile (25% weight)
      - Groundwater storage anomaly (20% weight)
      - Reservoir storage level (15% weight)
    """

    WEIGHTS = {
        "precipitation": 0.40,
        "soil_moisture":  0.25,
        "groundwater":    0.20,
        "reservoir":      0.15,
    }

    def compute(
        self,
        precip_anomaly_pct: float,   # % departure from normal (can be negative)
        sm_m3m3: float,              # Absolute soil moisture
        gws_anomaly_cm: float,       # GRACE-FO GWS anomaly
        reservoir_pct: float,        # Reservoir fill % (0-100)
    ) -> float:
        """
        Returns WAI score 0-100.
        100 = exceptional water availability
        0   = severe water crisis
        """
        # Normalize each component to 0-100
        precip_score = min(max(50.0 + precip_anomaly_pct * 0.5, 0.0), 100.0)
        sm_score = min(max(sm_m3m3 / 0.50 * 100.0, 0.0), 100.0)
        gws_score = min(max(50.0 + gws_anomaly_cm * 1.5, 0.0), 100.0)
        res_score = min(max(reservoir_pct, 0.0), 100.0)

        wai = (
            precip_score  * self.WEIGHTS["precipitation"] +
            sm_score      * self.WEIGHTS["soil_moisture"]  +
            gws_score     * self.WEIGHTS["groundwater"]    +
            res_score     * self.WEIGHTS["reservoir"]
        )
        return round(float(wai), 1)

    @staticmethod
    def classify(wai: float) -> WaterAvailabilityClass:
        if wai >= 80:  return WaterAvailabilityClass.ABUNDANT
        elif wai >= 60: return WaterAvailabilityClass.SUFFICIENT
        elif wai >= 40: return WaterAvailabilityClass.MODERATE
        elif wai >= 20: return WaitabilityClass.SCARCE
        return WaterAvailabilityClass.CRITICAL


# ---------------------------------------------------------------------------
# Planting recommendation engine
# ---------------------------------------------------------------------------

class PlantingAdvisor:
    """Generate planting recommendations based on WAI and season."""

    def recommend(
        self,
        wai: float,
        season: str,
        sm_m3m3: float,
        precip_3m_mm: float,
        irrigated: bool = True,
    ) -> tuple[PlantingRecommendation, str]:
        """
        Returns (recommendation, advisory_text).
        """
        if season == "MT-II" and not irrigated:
            # Rain-fed MT-II is highest risk
            if wai < 30:
                return (
                    PlantingRecommendation.SWITCH_TO_PALAWIJA,
                    (
                        f"WAI critically low ({wai:.0f}/100). Switch rain-fed paddy to "
                        f"short-season palawija (corn/soybean) to reduce crop failure risk. "
                        f"Estimated water deficit requires supplemental irrigation."
                    ),
                )
            elif wai < 50:
                return (
                    PlantingRecommendation.DELAY_4_WEEKS,
                    (
                        f"WAI below normal ({wai:.0f}/100). Delay planting by 4 weeks "
                        f"to wait for improved soil moisture. Apply mulching to conserve "
                        f"existing soil water ({sm_m3m3:.3f} m³/m³)."
                    ),
                )

        if wai >= 75:
            return (
                PlantingRecommendation.PROCEED_ON_SCHEDULE,
                (
                    f"Water availability excellent (WAI={wai:.0f}/100). Proceed on schedule. "
                    f"3-month cumulative precipitation forecast: {precip_3m_mm:.0f}mm."
                ),
            )
        elif wai >= 55:
            return (
                PlantingRecommendation.PROCEED_WITH_CAUTION,
                (
                    f"Water availability adequate but below normal (WAI={wai:.0f}/100). "
                    f"Proceed with water-efficient irrigation scheduling. "
                    f"Monitor reservoir levels and adjust water release schedule."
                ),
            )
        elif wai >= 40:
            return (
                PlantingRecommendation.DELAY_2_WEEKS,
                (
                    f"Water availability marginal (WAI={wai:.0f}/100). "
                    f"Delay planting by 2 weeks. "
                    f"Implement supplemental drip/sprinkler irrigation."
                ),
            )
        else:
            return (
                PlantingRecommendation.SUSPEND_RAIN_FED,
                (
                    f"Severe water deficit (WAI={wai:.0f}/100). Suspend rain-fed planting. "
                    f"Prioritize irrigated areas. Coordinate with PSDA for emergency water release."
                ),
            )


# ---------------------------------------------------------------------------
# Main seasonal forecast pipeline
# ---------------------------------------------------------------------------

class SeasonalWaterAvailabilityPipeline:
    """
    3-month seasonal water availability forecast pipeline.

    Runs monthly (Airflow DAG: seasonal_water_outlook_monthly).
    Ingests SMAP + GRACE-FO + QPE climatology + ENSO state →
    issues basin-level outlooks for Kementan agricultural planning.

    Usage:
        pipeline = SeasonalWaterAvailabilityPipeline()
        outlooks, status = pipeline.run(issued_date=date.today())
    """

    def __init__(self) -> None:
        self.enso = ENSOMonitor()
        self.wai_calc = WAICalculator()
        self.planting_advisor = PlantingAdvisor()
        logger.info(
            f"SeasonalWater pipeline initialized | "
            f"{len(AGRI_BASINS)} agricultural basins"
        )

    def run(
        self,
        issued_date: Optional[date] = None,
        smap_sm: Optional[dict] = None,       # {basin_id: sm_m3m3}
        grace_gws: Optional[dict] = None,     # {basin_id: gws_cm}
        reservoir_pct: Optional[dict] = None, # {basin_id: fill_%}
    ) -> tuple[list[SeasonalOutlook], SeasonalForecastStatus]:
        """Generate 3-month seasonal outlook for all agricultural basins."""
        run_time = datetime.utcnow()
        issued = issued_date or run_time.date()

        # Determine ENSO state
        enso_state = self.enso.get_current_state()
        logger.info(f"Seasonal water run | date={issued} ENSO={enso_state.value}")

        # Target months: next 3 months
        target_months = []
        for i in range(1, 4):
            m = (issued.month - 1 + i) % 12 + 1
            y = issued.year + ((issued.month - 1 + i) // 12)
            target_months.append((y, m))

        month_labels = [f"{y}-{m:02d}" for y, m in target_months]
        month_nums = [m for _, m in target_months]

        # Determine agricultural season
        season = "MT-I" if issued.month in PLANTING_SEASONS["MT-I"]["months"] else "MT-II"

        outlooks: list[SeasonalOutlook] = []

        for basin_id, basin_info in AGRI_BASINS.items():
            # Default inputs if not provided
            sm = (smap_sm or {}).get(basin_id, 0.28)
            gws = (grace_gws or {}).get(basin_id, -8.0)
            res = (reservoir_pct or {}).get(basin_id, 55.0)

            # Precipitation forecast
            precip_fc, precip_normal = self.enso.forecast_precip(
                basin_id, month_nums, enso_state
            )
            precip_anomaly = [
                round((fc / n - 1.0) * 100.0, 1) if n > 0 else 0.0
                for fc, n in zip(precip_fc, precip_normal)
            ]
            mean_anom = float(np.mean(precip_anomaly))

            # WAI
            wai = self.wai_calc.compute(
                precip_anomaly_pct=mean_anom,
                sm_m3m3=sm,
                gws_anomaly_cm=gws,
                reservoir_pct=res,
            )
            avail_class = WaterAvailabilityClass.CRITICAL
            if wai >= 80:   avail_class = WaterAvailabilityClass.ABUNDANT
            elif wai >= 60: avail_class = WaterAvailabilityClass.SUFFICIENT
            elif wai >= 40: avail_class = WaterAvailabilityClass.MODERATE
            elif wai >= 20: avail_class = WaterAvailabilityClass.SCARCE

            # Irrigation deficit (m³/ha/season)
            # Simplified: water gap = (crop requirement - effective rainfall) * area
            crop_req_mm = sum(PADDY_WATER_REQ_MM_DAY[s] * 30 for s in ["tillering", "heading"])
            eff_rainfall = sum(precip_fc) * 0.70  # 70% effective fraction
            deficit_mm = max(crop_req_mm - eff_rainfall, 0.0)
            irrigation_deficit_m3ha = deficit_mm * 10.0  # mm → m³/ha (1mm = 10m³/ha)

            # Planting recommendation
            precip_3m_total = sum(precip_fc)
            rec, advisory = self.planting_advisor.recommend(
                wai=wai,
                season=season,
                sm_m3m3=sm,
                precip_3m_mm=precip_3m_total,
                irrigated=(basin_info["irrigated_area_ha"] > 0),
            )

            # Confidence: higher in ENSO neutral, lower in El Niño
            conf_base = {"neutral": 82.0, "weak_el_nino": 72.0, "strong_el_nino": 60.0,
                        "weak_la_nina": 75.0, "strong_la_nina": 65.0}
            confidence = conf_base.get(enso_state.value, 70.0)

            outlook = SeasonalOutlook(
                basin_id=basin_id,
                basin_name=basin_info["name"],
                province=basin_info["province"],
                issued_date=issued,
                season=season,
                forecast_months=month_labels,
                wai_score=wai,
                availability_class=avail_class,
                precip_forecast_mm=[round(p, 1) for p in precip_fc],
                precip_normal_mm=[round(n, 1) for n in precip_normal],
                precip_anomaly_pct=precip_anomaly,
                soil_moisture_m3m3=round(sm, 3),
                groundwater_anomaly_cm=round(gws, 1),
                reservoir_storage_pct=round(res, 1),
                irrigation_deficit_m3ha=round(irrigation_deficit_m3ha, 0),
                planting_recommendation=rec,
                enso_state=enso_state,
                confidence_pct=round(confidence, 0),
                advisory=advisory,
            )
            outlooks.append(outlook)

            logger.info(
                f"{basin_id} | WAI={wai:.0f} [{avail_class.value}] "
                f"rec={rec.value} conf={confidence:.0f}%"
            )

        critical = [o for o in outlooks if o.availability_class == WaterAvailabilityClass.CRITICAL]
        scarce = [o for o in outlooks if o.availability_class == WaterAvailabilityClass.SCARCE]
        worst = min(outlooks, key=lambda o: o.wai_score) if outlooks else None

        status = SeasonalForecastStatus(
            run_time=run_time,
            issued_date=issued,
            n_basins=len(outlooks),
            n_critical_basins=len(critical),
            n_scarce_basins=len(scarce),
            most_critical_basin=worst.basin_name if worst else None,
            enso_state=enso_state.value,
        )

        logger.info(
            f"Seasonal water complete | basins={len(outlooks)} "
            f"critical={len(critical)} scarce={len(scarce)} "
            f"ENSO={enso_state.value}"
        )
        return outlooks, status
