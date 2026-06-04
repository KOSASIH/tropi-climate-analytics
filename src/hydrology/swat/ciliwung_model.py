"""
SWAT Hydrological Model — Ciliwung Watershed (Jakarta Flood Risk)
Agent: HYDROLOGIS | Tropi Climate Analytics

Ciliwung watershed: ~387 km², headwaters in Puncak (West Java) to Jakarta Bay.
Highest flood risk sub-watershed: Bogor–Depok–Jakarta corridor.

SWAT (Soil and Water Assessment Tool) simulates:
  - Surface runoff (CN2 / Green-Ampt infiltration)
  - Lateral flow and groundwater contribution
  - Evapotranspiration (Penman-Monteith)
  - Channel routing (Muskingum)
  - Sediment yield

HRU (Hydrological Response Unit) stratification:
  Land Cover × Soil Type × Slope class
  ~120 HRUs calibrated against BMKG Manggarai gauge (station ID: 196)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Watershed geometry constants
# ---------------------------------------------------------------------------

CILIWUNG_BBOX = {
    "lat_min": -6.65,
    "lat_max": -6.18,
    "lon_min": 106.70,
    "lon_max": 106.92,
}

CILIWUNG_AREA_KM2 = 387.0
CILIWUNG_OUTLET_LAT = -6.185  # Manggarai flood gate, Jakarta
CILIWUNG_OUTLET_LON = 106.850

# BMKG streamflow gauge — Manggarai
MANGGARAI_GAUGE_ID = "196"

# Alert thresholds (m above datum) — BNPB/BPBD DKI Jakarta
FLOOD_STAGE_SIAGA3 = 750   # cm  Warning
FLOOD_STAGE_SIAGA2 = 850   # cm  Alert
FLOOD_STAGE_SIAGA1 = 950   # cm  Emergency


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class LandCoverType(str, Enum):
    FOREST = "forest"
    DEGRADED_FOREST = "degraded_forest"
    PLANTATION = "plantation"
    CROPLAND = "cropland"
    URBAN = "urban"
    WATER = "water"


class SlopeClass(str, Enum):
    FLAT = "0-2%"
    GENTLE = "2-8%"
    MODERATE = "8-15%"
    STEEP = "15-30%"
    VERY_STEEP = ">30%"


@dataclass
class HRU:
    """Hydrological Response Unit."""
    hru_id: int
    land_cover: LandCoverType
    soil_type: str          # FAO soil classification
    slope_class: SlopeClass
    area_km2: float
    cn2: float              # SCS curve number (AMC II)
    esco: float = 0.95      # Soil evaporation compensation factor
    alpha_bf: float = 0.048 # Baseflow recession constant (day⁻¹)


@dataclass
class SWATRunoffResult:
    """SWAT water balance output for one time step."""
    timestamp: datetime
    precip_mm: float
    surface_runoff_mm: float
    lateral_flow_mm: float
    groundwater_mm: float
    et_mm: float            # Actual evapotranspiration
    soil_water_mm: float    # Soil water content
    streamflow_m3s: float   # Discharge at Manggarai gauge
    flood_stage_cm: float
    alert_level: Optional[str] = None  # siaga1/2/3/None


class SWATWaterBalance(BaseModel):
    """Aggregated SWAT water balance summary."""
    period_start: date
    period_end: date
    total_precip_mm: float
    total_runoff_mm: float
    total_et_mm: float
    total_gw_recharge_mm: float
    runoff_coefficient: float  # Q/P
    peak_discharge_m3s: float
    peak_discharge_date: date
    mean_daily_discharge_m3s: float


# ---------------------------------------------------------------------------
# SWAT parameter sets (calibrated for Ciliwung — Manggarai gauge 2010-2024)
# ---------------------------------------------------------------------------

DEFAULT_SWAT_PARAMS = {
    "cn2_forest": 55.0,
    "cn2_urban": 88.0,
    "cn2_cropland": 75.0,
    "cn2_plantation": 63.0,
    "smfmx": 4.5,       # Max snowmelt factor (not significant in tropics)
    "surlag": 4.0,      # Surface runoff lag (days)
    "gw_delay": 31.0,   # Groundwater delay (days)
    "gwqmn": 1.0,       # Threshold depth for baseflow (mm)
    "revapmn": 1.0,     # Threshold for revap from shallow aquifer
    "rchrg_dp": 0.05,   # Deep aquifer percolation fraction
    "alpha_bf": 0.048,  # Baseflow recession constant
    "ch_k2": 12.5,      # Channel hydraulic conductivity (mm/hr)
    "ch_n2": 0.014,     # Manning's n for main channel
    "slope_avg": 0.12,  # Average watershed slope (m/m)
    "slsubbsn": 60.0,   # Average slope length (m)
    "ov_n": 0.14,       # Manning's n for overland flow
}

# Calibrated HRUs for Ciliwung (representative subset)
CILIWUNG_HRUS: list[HRU] = [
    HRU(1, LandCoverType.FOREST,          "Andisol",   SlopeClass.STEEP,      45.2, 55.0),
    HRU(2, LandCoverType.DEGRADED_FOREST, "Andisol",   SlopeClass.MODERATE,   38.7, 67.0),
    HRU(3, LandCoverType.PLANTATION,      "Inceptisol",SlopeClass.GENTLE,     52.1, 63.0),
    HRU(4, LandCoverType.CROPLAND,        "Inceptisol",SlopeClass.FLAT,       68.4, 75.0),
    HRU(5, LandCoverType.URBAN,           "Entisol",   SlopeClass.FLAT,       98.3, 88.0),
    HRU(6, LandCoverType.URBAN,           "Entisol",   SlopeClass.GENTLE,     62.1, 86.0),
    HRU(7, LandCoverType.CROPLAND,        "Vertisol",  SlopeClass.GENTLE,     22.2, 78.0),
]


# ---------------------------------------------------------------------------
# Core SWAT engine
# ---------------------------------------------------------------------------

class CiliwungSWATModel:
    """
    SWAT-lite engine for Ciliwung watershed.

    Implements:
      - SCS-CN surface runoff estimation
      - Penman-Monteith ET (simplified)
      - Linear reservoir groundwater
      - Muskingum channel routing to Manggarai outlet

    Usage:
        model = CiliwungSWATModel()
        result = model.run_daily_step(
            precip_mm=45.0, tmax_c=32.0, tmin_c=24.0,
            solar_rad_mj=18.5, rh_pct=85.0, wind_ms=2.1
        )
    """

    def __init__(
        self,
        params: dict = None,
        hrus: list[HRU] = None,
    ) -> None:
        self.params = params or DEFAULT_SWAT_PARAMS
        self.hrus = hrus or CILIWUNG_HRUS
        self.total_area = sum(h.area_km2 for h in self.hrus)

        # State variables (mm)
        self._soil_water: float = 150.0      # Initial soil water
        self._shallow_aq: float = 200.0      # Shallow aquifer storage
        self._channel_storage: float = 0.0   # Channel water (Muskingum)

        logger.info(
            f"CiliwungSWAT initialized | {len(self.hrus)} HRUs | "
            f"Area: {self.total_area:.1f} km²"
        )

    # ------------------------------------------------------------------
    # SCS-CN runoff
    # ------------------------------------------------------------------

    def _scs_cn_runoff(self, precip_mm: float, cn2: float, soil_water_mm: float) -> float:
        """SCS curve number runoff (Q in mm)."""
        # AMC adjustment based on 5-day antecedent moisture
        cn = self._adjust_cn_for_moisture(cn2, soil_water_mm)
        s = (25400.0 / cn) - 254.0  # Potential maximum retention (mm)
        ia = 0.2 * s                 # Initial abstraction
        if precip_mm <= ia:
            return 0.0
        q = ((precip_mm - ia) ** 2) / (precip_mm - ia + s)
        return max(q, 0.0)

    def _adjust_cn_for_moisture(self, cn2: float, soil_water_mm: float) -> float:
        """Adjust CN from AMC-II to current antecedent moisture condition."""
        # Simplified Williams (1995) moisture-based CN adjustment
        ratio = min(max(soil_water_mm / 300.0, 0.0), 1.0)
        if ratio < 0.35:  # AMC I (dry)
            cn = cn2 * (4.2 * cn2) / (10.0 - 0.058 * cn2)
            return max(cn, 30.0)
        elif ratio > 0.70:  # AMC III (wet)
            cn = cn2 * (23.0 * cn2) / (10.0 + 0.13 * cn2)
            return min(cn, 99.0)
        return cn2

    # ------------------------------------------------------------------
    # Penman-Monteith ET (simplified FAO-56)
    # ------------------------------------------------------------------

    def _penman_monteith_et(
        self,
        tmax_c: float,
        tmin_c: float,
        solar_rad_mj: float,
        rh_pct: float,
        wind_ms: float,
        elevation_m: float = 250.0,
    ) -> float:
        """Reference ET₀ (mm/day) via FAO-56 Penman-Monteith."""
        import math
        tmean = (tmax_c + tmin_c) / 2.0
        # Slope of saturation vapor pressure curve
        delta = 4098.0 * (0.6108 * math.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
        # Atmospheric pressure
        P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
        gamma = 0.000665 * P  # Psychrometric constant
        # Saturation vapor pressure
        es = 0.5 * (0.6108 * math.exp(17.27 * tmax_c / (tmax_c + 237.3)) +
                    0.6108 * math.exp(17.27 * tmin_c / (tmin_c + 237.3)))
        ea = es * rh_pct / 100.0
        # Net radiation (simplified)
        rns = 0.77 * solar_rad_mj  # Net short-wave
        rnl = 0.5 * 4.903e-9 * ((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) * (
            0.34 - 0.14 * math.sqrt(ea)) * (1.35 * min(solar_rad_mj / 18.0, 1.0) - 0.35
        )
        rn = rns - rnl
        g = 0.0  # Soil heat flux (daily ≈ 0)
        # ET₀
        numerator = 0.408 * delta * (rn - g) + gamma * (900.0 / (tmean + 273.0)) * wind_ms * (es - ea)
        denominator = delta + gamma * (1.0 + 0.34 * wind_ms)
        return max(numerator / denominator, 0.0)

    # ------------------------------------------------------------------
    # Groundwater
    # ------------------------------------------------------------------

    def _groundwater_contribution(
        self,
        perc_mm: float,  # Percolation from soil profile
    ) -> float:
        """Linear reservoir baseflow contribution (mm/day)."""
        gw_delay = self.params["gw_delay"]
        alpha = self.params["alpha_bf"]
        w_seep = perc_mm * (1.0 - self.params["rchrg_dp"])
        self._shallow_aq += w_seep / gw_delay
        q_gw = alpha * self._shallow_aq
        self._shallow_aq = max(self._shallow_aq - q_gw, 0.0)
        return max(q_gw, 0.0) if self._shallow_aq > self.params["gwqmn"] else 0.0

    # ------------------------------------------------------------------
    # Channel routing (Muskingum)
    # ------------------------------------------------------------------

    def _muskingum_route(
        self,
        inflow_m3s: float,
        k: float = 0.5,   # Travel time (days)
        x: float = 0.2,   # Weighting factor
        dt: float = 1.0,  # Time step (days)
    ) -> float:
        """Muskingum channel routing — Ciliwung: Depok → Manggarai (~35 km)."""
        c0 = (dt - 2 * k * x) / (2 * k * (1 - x) + dt)
        c1 = (dt + 2 * k * x) / (2 * k * (1 - x) + dt)
        c2 = (2 * k * (1 - x) - dt) / (2 * k * (1 - x) + dt)
        outflow = c0 * inflow_m3s + c1 * self._channel_storage + c2 * max(self._channel_storage, 0)
        self._channel_storage = inflow_m3s
        return max(outflow, 0.0)

    # ------------------------------------------------------------------
    # Flood stage conversion
    # ------------------------------------------------------------------

    @staticmethod
    def discharge_to_stage(q_m3s: float) -> float:
        """
        Rating curve: Manggarai gauge (calibrated 2015-2024).
        h = a * Q^b  (power-law regression)
        a=82.4, b=0.43
        """
        if q_m3s <= 0:
            return 400.0  # Low-flow datum (cm)
        import math
        return 82.4 * (q_m3s ** 0.43)

    @staticmethod
    def stage_to_alert(stage_cm: float) -> Optional[str]:
        """Map stage to BPBD DKI Jakarta alert level."""
        if stage_cm >= FLOOD_STAGE_SIAGA1:
            return "siaga1"  # Emergency — evacuation
        elif stage_cm >= FLOOD_STAGE_SIAGA2:
            return "siaga2"  # Alert
        elif stage_cm >= FLOOD_STAGE_SIAGA3:
            return "siaga3"  # Warning
        return None

    # ------------------------------------------------------------------
    # Main daily time step
    # ------------------------------------------------------------------

    def run_daily_step(
        self,
        precip_mm: float,
        tmax_c: float,
        tmin_c: float,
        solar_rad_mj: float,
        rh_pct: float,
        wind_ms: float,
        timestamp: Optional[datetime] = None,
    ) -> SWATRunoffResult:
        """Run one daily SWAT time step across all HRUs → discharge at Manggarai."""
        ts = timestamp or datetime.utcnow()

        # Watershed-area-weighted runoff from all HRUs
        total_surface_q = 0.0
        for hru in self.hrus:
            wt = hru.area_km2 / self.total_area
            q = self._scs_cn_runoff(precip_mm, hru.cn2, self._soil_water)
            total_surface_q += wt * q

        # ET
        et0 = self._penman_monteith_et(tmax_c, tmin_c, solar_rad_mj, rh_pct, wind_ms)
        et_actual = min(et0, self._soil_water * 0.5)  # soil-limited

        # Soil water balance
        lateral_flow = max((self._soil_water - 250.0) * 0.05, 0.0)
        percolation = max((self._soil_water - 200.0) * 0.02, 0.0)
        self._soil_water += (precip_mm - total_surface_q - et_actual
                             - lateral_flow - percolation)
        self._soil_water = max(min(self._soil_water, 450.0), 0.0)

        # Groundwater baseflow
        gw_q = self._groundwater_contribution(percolation)

        # Total inflow to channel (m³/s)
        total_runoff_mm = total_surface_q + lateral_flow + gw_q
        q_m3s = (total_runoff_mm / 1000.0) * (self.total_area * 1e6) / 86400.0

        # Channel routing
        q_outlet = self._muskingum_route(q_m3s)

        # Stage and alert
        stage = self.discharge_to_stage(q_outlet)
        alert = self.stage_to_alert(stage)

        if alert:
            logger.warning(
                f"CILIWUNG FLOOD ALERT [{alert.upper()}] "
                f"Stage={stage:.0f}cm Q={q_outlet:.1f}m³/s @ {ts.isoformat()}"
            )

        return SWATRunoffResult(
            timestamp=ts,
            precip_mm=precip_mm,
            surface_runoff_mm=round(total_surface_q, 2),
            lateral_flow_mm=round(lateral_flow, 2),
            groundwater_mm=round(gw_q, 2),
            et_mm=round(et_actual, 2),
            soil_water_mm=round(self._soil_water, 2),
            streamflow_m3s=round(q_outlet, 2),
            flood_stage_cm=round(stage, 1),
            alert_level=alert,
        )

    # ------------------------------------------------------------------
    # Multi-step simulation
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        weather_records: list[dict],
    ) -> list[SWATRunoffResult]:
        """
        Run SWAT over a list of daily weather records.

        Each record: {timestamp, precip_mm, tmax_c, tmin_c,
                      solar_rad_mj, rh_pct, wind_ms}
        """
        results = []
        for rec in weather_records:
            results.append(self.run_daily_step(**rec))
        logger.info(f"SWAT simulation complete: {len(results)} days")
        return results

    def compute_water_balance(
        self,
        results: list[SWATRunoffResult],
    ) -> SWATWaterBalance:
        """Aggregate simulation results into water balance summary."""
        if not results:
            raise ValueError("Empty simulation results")
        total_p = sum(r.precip_mm for r in results)
        total_q = sum(r.surface_runoff_mm + r.lateral_flow_mm + r.groundwater_mm for r in results)
        total_et = sum(r.et_mm for r in results)
        total_gw = sum(r.groundwater_mm for r in results)
        peak_r = max(results, key=lambda r: r.streamflow_m3s)
        return SWATWaterBalance(
            period_start=results[0].timestamp.date(),
            period_end=results[-1].timestamp.date(),
            total_precip_mm=round(total_p, 1),
            total_runoff_mm=round(total_q, 1),
            total_et_mm=round(total_et, 1),
            total_gw_recharge_mm=round(total_gw, 1),
            runoff_coefficient=round(total_q / total_p, 3) if total_p > 0 else 0.0,
            peak_discharge_m3s=round(peak_r.streamflow_m3s, 2),
            peak_discharge_date=peak_r.timestamp.date(),
            mean_daily_discharge_m3s=round(
                sum(r.streamflow_m3s for r in results) / len(results), 2
            ),
        )
