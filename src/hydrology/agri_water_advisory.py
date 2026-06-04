"""
Agricultural Water Advisory — Sprint 3 Deliverable 3
HYDROLOGIS | Tropi Climate Analytics

Seasonal (30/60/90-day) irrigation water availability advisory for agricultural zones.

Inputs:
  - seasonal_water_availability.py outputs (Sprint 2) — WAI composite per province
  - SMAP root-zone soil moisture (L4 daily)
  - BMKG seasonal rainfall outlook (ENSO phase from SeasonalWaterAvailabilityPipeline)

Output:
  - AgriWaterAdvisory Pydantic model per watershed
  - Prometheus gauge: tropi_agri_water_advisory_class{watershed_id}
    Values: 0=SURPLUS, 1=ADEQUATE, 2=DEFICIT, 3=CRITICAL

Advisory classes: SURPLUS / ADEQUATE / DEFICIT / CRITICAL
Crop calendar aligned to Kementan MT I (Oct-Mar) / MT II (Apr-Sep) seasons.

Watershed scope: 38 national strategic watersheds (DAS Strategis Nasional per
  Kep. MenLHK P.10/2019).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE  = os.getenv("WORKSPACE_ROOT", "workspace")
OUTPUT_DIR = os.path.join(WORKSPACE, "output", "agri_advisory")

# Advisory class numeric codes for Prometheus gauge
ADVISORY_CLASS_CODE = {
    "SURPLUS":  0,
    "ADEQUATE": 1,
    "DEFICIT":  2,
    "CRITICAL": 3,
}

# 38 DAS Strategis Nasional (abbreviated set — production: full PermenLHK annex)
STRATEGIC_WATERSHEDS: dict[str, dict] = {
    "das_ciliwung":        {"name": "DAS Ciliwung",        "province": "DKI Jakarta / Jawa Barat", "irrigated_ha": 12000, "primary_crop": "padi"},
    "das_citarum":         {"name": "DAS Citarum",         "province": "Jawa Barat",               "irrigated_ha": 225000, "primary_crop": "padi"},
    "das_cisanggarung":    {"name": "DAS Cisanggarung",    "province": "Jawa Barat / Jawa Tengah", "irrigated_ha": 35000, "primary_crop": "padi"},
    "das_serayu":          {"name": "DAS Serayu",          "province": "Jawa Tengah",              "irrigated_ha": 78000, "primary_crop": "padi"},
    "das_bengawan_solo":   {"name": "DAS Bengawan Solo",   "province": "Jawa Tengah / Jawa Timur", "irrigated_ha": 135000, "primary_crop": "padi"},
    "das_brantas":         {"name": "DAS Brantas",         "province": "Jawa Timur",               "irrigated_ha": 145000, "primary_crop": "padi/tebu"},
    "das_musi":            {"name": "DAS Musi",            "province": "Sumatera Selatan",         "irrigated_ha": 89000, "primary_crop": "padi/sawit"},
    "das_batang_hari":     {"name": "DAS Batang Hari",     "province": "Jambi / Sumatera Barat",   "irrigated_ha": 42000, "primary_crop": "padi"},
    "das_kampar":          {"name": "DAS Kampar",          "province": "Riau",                     "irrigated_ha": 18000, "primary_crop": "sawit"},
    "das_kapuas":          {"name": "DAS Kapuas",          "province": "Kalimantan Barat",         "irrigated_ha": 62000, "primary_crop": "padi"},
    "das_barito":          {"name": "DAS Barito",          "province": "Kalimantan Tengah/Selatan","irrigated_ha": 55000, "primary_crop": "padi/sawit"},
    "das_mahakam":         {"name": "DAS Mahakam",         "province": "Kalimantan Timur",         "irrigated_ha": 28000, "primary_crop": "padi"},
    "das_jeneberang":      {"name": "DAS Jeneberang",      "province": "Sulawesi Selatan",         "irrigated_ha": 35000, "primary_crop": "padi"},
    "das_saddang":         {"name": "DAS Saddang",         "province": "Sulawesi Selatan",         "irrigated_ha": 30000, "primary_crop": "padi"},
    "das_memberamo":       {"name": "DAS Memberamo",       "province": "Papua",                    "irrigated_ha": 8000,  "primary_crop": "padi"},
    "das_digul":           {"name": "DAS Digul",           "province": "Papua Selatan",            "irrigated_ha": 5000,  "primary_crop": "padi"},
    "das_progo_opak":      {"name": "DAS Progo-Opak-Oyo",  "province": "DI Yogyakarta",           "irrigated_ha": 47000, "primary_crop": "padi"},
    "das_pemali_comal":    {"name": "DAS Pemali-Comal",    "province": "Jawa Tengah",              "irrigated_ha": 65000, "primary_crop": "padi"},
    "das_toba_asahan":     {"name": "DAS Toba-Asahan",     "province": "Sumatera Utara",           "irrigated_ha": 56000, "primary_crop": "padi/sawit"},
    "das_akucem":          {"name": "DAS Aek Ucim",        "province": "Sumatera Utara",           "irrigated_ha": 12000, "primary_crop": "padi"},
}

# Kementan planting season calendar
PRIMARY_PLANTING_MONTHS   = {10, 11, 12, 1, 2, 3}   # MT I (musim tanam 1)
SECONDARY_PLANTING_MONTHS = {4, 5, 6, 7, 8, 9}       # MT II

# ---------------------------------------------------------------------------
# Prometheus gauge
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Gauge
    from src.hydrology.metrics import _REGISTRY

    AGRI_ADVISORY_CLASS_GAUGE = Gauge(
        "tropi_agri_water_advisory_class",
        "Agricultural water advisory class: 0=SURPLUS 1=ADEQUATE 2=DEFICIT 3=CRITICAL",
        labelnames=["watershed_id"],
        registry=_REGISTRY,
    )
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False
    class _StubGauge:
        def labels(self, **_): return self
        def set(self, *_): pass
    AGRI_ADVISORY_CLASS_GAUGE = _StubGauge()  # type: ignore[assignment]


def _record_advisory_class(watershed_id: str, class_str: str) -> None:
    code = ADVISORY_CLASS_CODE.get(class_str, 1)
    AGRI_ADVISORY_CLASS_GAUGE.labels(watershed_id=watershed_id).set(code)


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class WaterAvailabilityClass(str, Enum):
    SURPLUS  = "SURPLUS"
    ADEQUATE = "ADEQUATE"
    DEFICIT  = "DEFICIT"
    CRITICAL = "CRITICAL"


class AgriWaterAdvisory(BaseModel):
    watershed_id:            str
    watershed_name:          str
    province:                str
    primary_crop:            str
    advisory_horizon_days:   int
    water_availability_class: WaterAvailabilityClass
    wai_score:               float    = Field(..., description="Water Availability Index (-3 to +3)")
    reservoir_fill_pct:      float    = Field(..., description="Estimated reservoir fill (%)")
    sm_rootzone_m3m3:        float    = Field(..., description="SMAP root-zone soil moisture")
    irrigation_recommendation: str
    crop_calendar_flag:      str      = Field(
        ..., description="MT_I_PROCEED / MT_I_DELAY / MT_II_PROCEED / MT_II_DELAY / HOLD"
    )
    enso_phase:              str
    irrigated_area_ha:       int
    issued_at:               datetime


class AgriAdvisoryRunStatus(BaseModel):
    run_time_utc:           datetime
    forecast_month:         date
    enso_phase:             str
    watersheds_processed:   int
    watersheds_critical:    list[str]
    watersheds_deficit:     list[str]
    output_path:            str
    success:                bool
    error:                  Optional[str] = None


# ---------------------------------------------------------------------------
# Irrigation recommendation engine
# ---------------------------------------------------------------------------

_CRITICAL_THRESHOLDS = {
    "wai_critical":  -1.5,    # WAI < -1.5 → CRITICAL
    "wai_deficit":   -0.5,    # WAI ∈ [-1.5, -0.5] → DEFICIT
    "wai_surplus":    0.5,    # WAI > 0.5 → SURPLUS
    "reservoir_critical": 20.0,  # Fill % < 20 → critical
    "reservoir_deficit":  40.0,
    "sm_wilting":    0.15,    # m³/m³ — near wilting point
}


def _classify_water_availability(
    wai: float,
    reservoir_pct: float,
    sm_m3m3: float,
) -> WaterAvailabilityClass:
    # Critical if any of: WAI very low, reservoir near empty, or SM near wilting
    if (wai < _CRITICAL_THRESHOLDS["wai_critical"]
            or reservoir_pct < _CRITICAL_THRESHOLDS["reservoir_critical"]
            or sm_m3m3 < _CRITICAL_THRESHOLDS["sm_wilting"]):
        return WaterAvailabilityClass.CRITICAL
    if wai < _CRITICAL_THRESHOLDS["wai_deficit"] or reservoir_pct < _CRITICAL_THRESHOLDS["reservoir_deficit"]:
        return WaterAvailabilityClass.DEFICIT
    if wai > _CRITICAL_THRESHOLDS["wai_surplus"]:
        return WaterAvailabilityClass.SURPLUS
    return WaterAvailabilityClass.ADEQUATE


def _build_irrigation_recommendation(
    avail_class: WaterAvailabilityClass,
    reservoir_pct: float,
    irrigated_ha: int,
    enso_phase: str,
) -> str:
    if avail_class == WaterAvailabilityClass.CRITICAL:
        return (
            f"SIAGA AIR: Pengisian bendungan {reservoir_pct:.0f}%. "
            "Terapkan irigasi tetes/tetes bergilir. Kurangi luas tanam 40-50% "
            "dari {irrigated_ha:,} ha. Koordinasi BPSDA untuk alokasi darurat."
        )
    if avail_class == WaterAvailabilityClass.DEFICIT:
        return (
            f"Ketersediaan air di bawah normal (bendungan {reservoir_pct:.0f}%). "
            "Terapkan SRI/AWD (alternate wetting and drying). "
            "Jadwalkan giliran irigasi; prioritaskan fase generatif padi."
        )
    if avail_class == WaterAvailabilityClass.SURPLUS:
        if enso_phase == "LA_NINA":
            return (
                "Curah hujan di atas normal (La Niña aktif). "
                "Manfaatkan surplus untuk pengisian embung/reservoir. "
                "Waspadai ancaman banjir lahan; siapkan saluran drainase."
            )
        return (
            f"Ketersediaan air memadai (bendungan {reservoir_pct:.0f}%). "
            "Lanjutkan jadwal irigasi normal. Pertimbangkan perluasan tanam {irrigated_ha:,} ha."
        )
    return (
        f"Ketersediaan air normal (bendungan {reservoir_pct:.0f}%). "
        "Jalankan jadwal irigasi sesuai RTTG. Pantau perkembangan ENSO tiap 2 minggu."
    )


def _crop_calendar_flag(
    avail_class: WaterAvailabilityClass,
    forecast_month: date,
) -> str:
    month = forecast_month.month
    season = "MT_I" if month in PRIMARY_PLANTING_MONTHS else "MT_II"
    if avail_class == WaterAvailabilityClass.CRITICAL:
        return "HOLD"
    if avail_class == WaterAvailabilityClass.DEFICIT:
        return f"{season}_DELAY"
    return f"{season}_PROCEED"


# ---------------------------------------------------------------------------
# Advisory generator
# ---------------------------------------------------------------------------

class AgriWaterAdvisoryGenerator:
    """
    Generates seasonal agricultural water advisories for 20 strategic watersheds.

    Usage (from Airflow seasonal_monthly DAG):
        gen = AgriWaterAdvisoryGenerator()
        status = gen.run(forecast_month=date(2026, 7, 1))
    """

    def __init__(self) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run(
        self,
        forecast_month: date,
        advisory_horizons: Optional[list[int]] = None,
    ) -> AgriAdvisoryRunStatus:
        horizons = advisory_horizons or [30, 60, 90]
        run_time = datetime.now(timezone.utc)

        # Pull seasonal water availability outputs (Sprint 2)
        wai_data = self._load_wai_data(forecast_month)
        enso_phase = wai_data.get("enso_phase", "NEUTRAL")

        advisories: list[AgriWaterAdvisory] = []
        for ws_id, ws_meta in STRATEGIC_WATERSHEDS.items():
            # Derive WAI from Sprint 2 province-level output (nearest province)
            wai = self._lookup_wai(ws_id, wai_data)
            # SMAP root-zone SM (stub — production: join with SMAP L4)
            sm = self._fetch_sm_rootzone(ws_id, forecast_month)
            # Reservoir fill % (stub — production: BBWS OTOKLIM API)
            reservoir_pct = self._fetch_reservoir_fill(ws_id, wai, sm)

            for horizon in horizons:
                avail_class = _classify_water_availability(wai, reservoir_pct, sm)
                irr_rec = _build_irrigation_recommendation(
                    avail_class, reservoir_pct, ws_meta["irrigated_ha"], enso_phase
                )
                flag = _crop_calendar_flag(avail_class, forecast_month)

                advisory = AgriWaterAdvisory(
                    watershed_id=ws_id,
                    watershed_name=ws_meta["name"],
                    province=ws_meta["province"],
                    primary_crop=ws_meta["primary_crop"],
                    advisory_horizon_days=horizon,
                    water_availability_class=avail_class,
                    wai_score=round(wai, 3),
                    reservoir_fill_pct=round(reservoir_pct, 1),
                    sm_rootzone_m3m3=round(sm, 4),
                    irrigation_recommendation=irr_rec,
                    crop_calendar_flag=flag,
                    enso_phase=enso_phase,
                    irrigated_area_ha=ws_meta["irrigated_ha"],
                    issued_at=run_time,
                )
                advisories.append(advisory)
                # Emit gauge (use 30-day horizon as canonical label)
                if horizon == 30:
                    _record_advisory_class(ws_id, avail_class.value)

        critical = [a.watershed_name for a in advisories
                    if a.water_availability_class == WaterAvailabilityClass.CRITICAL and a.advisory_horizon_days == 30]
        deficit  = [a.watershed_name for a in advisories
                    if a.water_availability_class == WaterAvailabilityClass.DEFICIT and a.advisory_horizon_days == 30]

        output_path = self._write_output(forecast_month, advisories, enso_phase)

        logger.info(
            "Agri advisory complete | month=%s critical=%d deficit=%d",
            forecast_month, len(critical), len(deficit),
        )

        return AgriAdvisoryRunStatus(
            run_time_utc=run_time,
            forecast_month=forecast_month,
            enso_phase=enso_phase,
            watersheds_processed=len(STRATEGIC_WATERSHEDS),
            watersheds_critical=critical,
            watersheds_deficit=deficit,
            output_path=output_path,
            success=True,
        )

    # ------------------------------------------------------------------
    # Data fetchers (stubs for Sprint 3 — production wired in Sprint 4)
    # ------------------------------------------------------------------

    def _load_wai_data(self, forecast_month: date) -> dict:
        """Load Sprint 2 seasonal water availability JSON."""
        path = os.path.join(
            WORKSPACE, "output", "seasonal",
            f"water_availability_advisory_{forecast_month.strftime('%Y%m')}.json"
        )
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        # Fallback: neutral ENSO
        return {"enso_phase": "NEUTRAL", "provinces": []}

    def _lookup_wai(self, ws_id: str, wai_data: dict) -> float:
        """Map watershed → province WAI score from Sprint 2 output."""
        provinces = wai_data.get("provinces", [])
        # Watershed→province heuristic mapping
        ws_province_map = {
            "das_ciliwung":      "DKI Jakarta",
            "das_citarum":       "Jawa Barat",
            "das_serayu":        "Jawa Tengah",
            "das_bengawan_solo": "Jawa Tengah",
            "das_brantas":       "Jawa Timur",
            "das_musi":          "Sumatera Selatan",
            "das_batang_hari":   "Jambi",
            "das_kapuas":        "Kalimantan Barat",
            "das_barito":        "Kalimantan Tengah",
            "das_mahakam":       "Kalimantan Timur",
            "das_jeneberang":    "Sulawesi Selatan",
        }
        target_prov = ws_province_map.get(ws_id)
        for p in provinces:
            if target_prov and target_prov in p.get("province_name", ""):
                return float(p.get("wai_composite", 0.0))
        # Default: climatological neutral
        rng = np.random.default_rng(hash(ws_id) % (2**32))
        return float(np.clip(rng.normal(0.0, 0.6), -2.5, 2.5))

    def _fetch_sm_rootzone(self, ws_id: str, forecast_month: date) -> float:
        """Stub: return SMAP root-zone SM for watershed centroid."""
        rng = np.random.default_rng(hash(ws_id) % (2**32) + forecast_month.toordinal())
        return float(np.clip(rng.normal(0.28, 0.06), 0.10, 0.50))

    def _fetch_reservoir_fill(self, ws_id: str, wai: float, sm: float) -> float:
        """Stub: estimate reservoir fill % from WAI + SM proxy."""
        base = 55.0 + wai * 15.0 + (sm - 0.28) * 80.0
        rng = np.random.default_rng(hash(ws_id + "res") % (2**32))
        fill = base + float(rng.normal(0, 8))
        return float(np.clip(fill, 5.0, 100.0))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_output(
        self,
        forecast_month: date,
        advisories: list[AgriWaterAdvisory],
        enso_phase: str,
    ) -> str:
        fname = f"agri_water_advisory_{forecast_month.strftime('%Y%m')}.json"
        path  = os.path.join(OUTPUT_DIR, fname)
        payload = {
            "title": "Prakiraan Ketersediaan Air Irigasi — HYDROLOGIS Sprint 3",
            "forecast_month": forecast_month.isoformat(),
            "enso_phase": enso_phase,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "sprint3.0",
            "advisories": [a.model_dump() for a in advisories],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Agri advisory written: %s", path)
        return path
