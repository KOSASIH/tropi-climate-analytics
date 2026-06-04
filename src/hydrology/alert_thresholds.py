"""
Siaga 1/2/3 alert threshold table for BNPB flood early warning.

Covers all 3 monitored river stations:
  - Ciliwung / Pos Manggarai   (DKI Jakarta)
  - Brantas  / Pos Mlirip      (Jawa Timur)
  - Solo     / Pos Jurug       (Jawa Tengah)

Thresholds are water stage (cm above staff gauge datum) mapped to BNPB
Siaga levels. Calibrated from:
  - BBWS field surveys (2015-2024)
  - Annual exceedance probability derived from GEV distribution fit (L-moments)
  - Return period analysis (2 / 5 / 10 / 25 / 50 yr)

Usage:
    from src.hydrology.alert_thresholds import ALERT_THRESHOLDS, get_siaga_level

    level = get_siaga_level("ciliwung", stage_cm=850)
    # → FloodAlertLevel.SIAGA_2
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Siaga level enum (mirrors FloodAlertLevel in flood_prediction.py)
# ---------------------------------------------------------------------------

class SiagaLevel(str, Enum):
    NORMAL  = "NORMAL"
    SIAGA_3 = "SIAGA_3"   # Watch   — elevated stage, monitor closely
    SIAGA_2 = "SIAGA_2"   # Warning — inundation imminent in low-lying areas
    SIAGA_1 = "SIAGA_1"   # Critical — active flooding, evacuate


# ---------------------------------------------------------------------------
# Threshold record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageThreshold:
    """Stage threshold for one Siaga transition at a given station."""
    stage_cm: float                     # Water level at staff gauge (cm above datum)
    return_period_yr: float             # Associated return period (years)
    exceedance_prob_annual: float       # Annual exceedance probability (AEP)
    discharge_m3s: float                # Approximate discharge at threshold (m³/s)
    calibration_source: str             # Data source / survey reference
    calibration_year: int


@dataclass(frozen=True)
class StationThresholds:
    """Complete Siaga threshold set for one gauging station."""
    station_id: str
    station_name: str
    river: str
    province: str
    latitude: float
    longitude: float
    datum_masl: float                   # Gauge datum elevation (metres above sea level)
    bankfull_stage_cm: float            # Stage at bankfull discharge
    bankfull_q_m3s: float               # Bankfull discharge (m³/s)

    # Stage thresholds (cm above gauge datum)
    siaga_3: StageThreshold             # Watch threshold
    siaga_2: StageThreshold             # Warning threshold
    siaga_1: StageThreshold             # Critical / evacuate threshold

    # GEV distribution parameters (shape ξ, location μ, scale σ) — annual maxima
    gev_shape:    float
    gev_location: float
    gev_scale:    float

    # Historical record
    record_period_start: int            # Year of earliest reliable discharge record
    record_period_end:   int
    max_observed_stage_cm: float
    max_observed_year:  int
    notes: str


# ---------------------------------------------------------------------------
# Station threshold table
# ---------------------------------------------------------------------------

ALERT_THRESHOLDS: dict[str, StationThresholds] = {

    # ── Ciliwung / Pos Manggarai ─────────────────────────────────────────
    "ciliwung": StationThresholds(
        station_id="ID-CIL-MANG",
        station_name="Pos Manggarai",
        river="Ciliwung",
        province="DKI Jakarta",
        latitude=-6.2088,
        longitude=106.8456,
        datum_masl=4.7,
        bankfull_stage_cm=750.0,
        bankfull_q_m3s=250.0,

        siaga_3=StageThreshold(
            stage_cm=750.0,
            return_period_yr=1.5,
            exceedance_prob_annual=0.667,
            discharge_m3s=250.0,
            calibration_source="BBWS Ciliwung-Cisadane / BNPB Rating Curve 2022",
            calibration_year=2022,
        ),
        siaga_2=StageThreshold(
            stage_cm=850.0,
            return_period_yr=5.0,
            exceedance_prob_annual=0.200,
            discharge_m3s=380.0,
            calibration_source="BBWS Ciliwung-Cisadane / BNPB Rating Curve 2022",
            calibration_year=2022,
        ),
        siaga_1=StageThreshold(
            stage_cm=950.0,
            return_period_yr=25.0,
            exceedance_prob_annual=0.040,
            discharge_m3s=580.0,
            calibration_source="BBWS Ciliwung-Cisadane / Flood Hazard Study 2020",
            calibration_year=2020,
        ),

        # GEV (annual maxima, 1985-2024, n=40 station-years)
        gev_shape=-0.12,
        gev_location=810.0,
        gev_scale=95.0,

        record_period_start=1985,
        record_period_end=2024,
        max_observed_stage_cm=1090.0,
        max_observed_year=2020,
        notes=(
            "Manggarai benchmark station for Jakarta flood watch. "
            "Siaga 1 triggers Jatiluhur reservoir spill coordination. "
            "Tidal backwater influence below stage 600 cm (MW=+0.3 m). "
            "2020 flood peak (1090 cm) exceeded 50-year return level."
        ),
    ),

    # ── Brantas / Pos Mlirip ────────────────────────────────────────────
    "brantas": StationThresholds(
        station_id="ID-BRT-MLRP",
        station_name="Pos Mlirip",
        river="Brantas",
        province="Jawa Timur",
        latitude=-7.3842,
        longitude=112.5607,
        datum_masl=12.1,
        bankfull_stage_cm=420.0,
        bankfull_q_m3s=1800.0,

        siaga_3=StageThreshold(
            stage_cm=420.0,
            return_period_yr=2.0,
            exceedance_prob_annual=0.500,
            discharge_m3s=1800.0,
            calibration_source="BBWS Brantas / PUSAIR Rating Curve 2021",
            calibration_year=2021,
        ),
        siaga_2=StageThreshold(
            stage_cm=500.0,
            return_period_yr=10.0,
            exceedance_prob_annual=0.100,
            discharge_m3s=2600.0,
            calibration_source="BBWS Brantas / PUSAIR Rating Curve 2021",
            calibration_year=2021,
        ),
        siaga_1=StageThreshold(
            stage_cm=590.0,
            return_period_yr=50.0,
            exceedance_prob_annual=0.020,
            discharge_m3s=3500.0,
            calibration_source="BBWS Brantas / Regional Flood Frequency 2019",
            calibration_year=2019,
        ),

        # GEV (1978-2024, n=47 station-years)
        gev_shape=-0.08,
        gev_location=455.0,
        gev_scale=75.0,

        record_period_start=1978,
        record_period_end=2024,
        max_observed_stage_cm=640.0,
        max_observed_year=2007,
        notes=(
            "Mlirip controls Brantas delta bifurcation (Kali Mas / Kali Porong). "
            "Siaga 1 activates Porong emergency spillway operations. "
            "Lapindo mud-flow legacy affects lower Porong channel hydraulics. "
            "Stage-discharge rating re-surveyed 2021 post-channel dredging."
        ),
    ),

    # ── Solo (Bengawan Solo) / Pos Jurug ────────────────────────────────
    "solo": StationThresholds(
        station_id="ID-SOL-JURG",
        station_name="Pos Jurug",
        river="Bengawan Solo",
        province="Jawa Tengah",
        latitude=-7.5667,
        longitude=110.8569,
        datum_masl=92.4,
        bankfull_stage_cm=550.0,
        bankfull_q_m3s=2500.0,

        siaga_3=StageThreshold(
            stage_cm=550.0,
            return_period_yr=2.0,
            exceedance_prob_annual=0.500,
            discharge_m3s=2500.0,
            calibration_source="BBWS Bengawan Solo / PJT I Rating Curve 2023",
            calibration_year=2023,
        ),
        siaga_2=StageThreshold(
            stage_cm=650.0,
            return_period_yr=10.0,
            exceedance_prob_annual=0.100,
            discharge_m3s=3600.0,
            calibration_source="BBWS Bengawan Solo / PJT I Rating Curve 2023",
            calibration_year=2023,
        ),
        siaga_1=StageThreshold(
            stage_cm=750.0,
            return_period_yr=25.0,
            exceedance_prob_annual=0.040,
            discharge_m3s=4800.0,
            calibration_source="BBWS Bengawan Solo / Regional Flood Frequency 2018",
            calibration_year=2018,
        ),

        # GEV (1971-2024, n=54 station-years — longest record in dataset)
        gev_shape=-0.15,
        gev_location=610.0,
        gev_scale=110.0,

        record_period_start=1971,
        record_period_end=2024,
        max_observed_stage_cm=870.0,
        max_observed_year=2010,
        notes=(
            "Longest Solo River continuous record; UNESCO reference station. "
            "Jurug is the primary upstream forecast control point for "
            "Solo delta (Demak / Gresik lowlands, pop. ~1.2M at risk). "
            "2010 flood (870 cm, ~100-yr event) caused 79 fatalities, IDR 2.9T damage. "
            "Bengawan Solo slow-rise: Siaga 1 → inundation lag ~18-24 h downstream. "
            "GEV calibrated with PWM L-moments (Hosking & Wallis 1997)."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------

def get_siaga_level(river_id: str, stage_cm: float) -> SiagaLevel:
    """
    Classify water stage into BNPB Siaga level for a given river station.

    Args:
        river_id:  One of "ciliwung", "brantas", "solo".
        stage_cm:  Current water stage in cm above gauge datum.

    Returns:
        SiagaLevel (NORMAL / SIAGA_3 / SIAGA_2 / SIAGA_1).

    Raises:
        KeyError: If river_id is not in ALERT_THRESHOLDS.
    """
    thresholds = ALERT_THRESHOLDS[river_id]
    if stage_cm >= thresholds.siaga_1.stage_cm:
        return SiagaLevel.SIAGA_1
    if stage_cm >= thresholds.siaga_2.stage_cm:
        return SiagaLevel.SIAGA_2
    if stage_cm >= thresholds.siaga_3.stage_cm:
        return SiagaLevel.SIAGA_3
    return SiagaLevel.NORMAL


def get_exceedance_probability(river_id: str, stage_cm: float) -> float:
    """
    Estimate annual exceedance probability for an arbitrary stage value
    using the fitted GEV distribution.

    Args:
        river_id: One of "ciliwung", "brantas", "solo".
        stage_cm: Target stage in cm.

    Returns:
        Annual exceedance probability (0.0 – 1.0).
    """
    from scipy.stats import genextreme  # type: ignore[import]
    t = ALERT_THRESHOLDS[river_id]
    # scipy GEV uses sign-reversed shape convention
    cdf = genextreme.cdf(
        stage_cm,
        c=-t.gev_shape,
        loc=t.gev_location,
        scale=t.gev_scale,
    )
    return float(max(0.0, min(1.0, 1.0 - cdf)))


def return_period_for_stage(river_id: str, stage_cm: float) -> float:
    """Return the estimated return period (years) for a given stage."""
    aep = get_exceedance_probability(river_id, stage_cm)
    return 1.0 / aep if aep > 0 else float("inf")


# ---------------------------------------------------------------------------
# Summary table (human-readable)
# ---------------------------------------------------------------------------

THRESHOLD_SUMMARY: list[dict] = [
    {
        "river_id":    rid,
        "station":     t.station_name,
        "river":       t.river,
        "province":    t.province,
        "siaga_3_cm":  t.siaga_3.stage_cm,
        "siaga_3_rp":  t.siaga_3.return_period_yr,
        "siaga_3_q":   t.siaga_3.discharge_m3s,
        "siaga_2_cm":  t.siaga_2.stage_cm,
        "siaga_2_rp":  t.siaga_2.return_period_yr,
        "siaga_2_q":   t.siaga_2.discharge_m3s,
        "siaga_1_cm":  t.siaga_1.stage_cm,
        "siaga_1_rp":  t.siaga_1.return_period_yr,
        "siaga_1_q":   t.siaga_1.discharge_m3s,
        "record_max_stage_cm": t.max_observed_stage_cm,
        "record_max_year":     t.max_observed_year,
    }
    for rid, t in ALERT_THRESHOLDS.items()
]
