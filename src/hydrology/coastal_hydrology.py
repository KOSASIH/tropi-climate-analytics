"""
coastal_hydrology.py — Sprint 11 P5
CoastalHydrology: Tidal–river discharge interaction and saltwater intrusion for 5 cities.

Cities: Jakarta, Semarang, Surabaya, Demak, Pekalongan

Methods:
  compute_saltwater_intrusion(city_id, date, river_discharge_cms, tide_level_m) → IntrusionResult
  compute_tidal_backwater(river_id, date, discharge_cms)                         → TidalBackwaterResult
  get_coastal_flood_risk(city_id, intrusion, backwater)                         → CoastalFloodRisk

SII = (Q_tidal / Q_river) × (1 + relative_sea_level_anomaly_m)
Backwater extension (km) = k × (tide_height_m / slope) × (1 / sqrt(discharge))

Outputs:
  workspace/output/coastal/coastal_{city_id}_{YYYYMMDD}.json
  workspace/output/coastal/active_coastal_risk.json

Prometheus: SALTWATER_INTRUSION_INDEX{city_id} Gauge
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ALL_CITIES = ["jakarta", "semarang", "surabaya", "demak", "pekalongan"]

# SII categories
_SII_CATEGORIES = [
    ("CRITICAL",  1.20),
    ("HIGH",      0.70),
    ("MODERATE",  0.30),
    ("SAFE",      0.00),
]

# River channel slope (m/km) — approximate
_RIVER_SLOPE: dict[str, float] = {
    "ciliwung":          0.00040,
    "banjir_kanal_barat":0.00035,
    "banjir_kanal_timur":0.00035,
    "kali_semarang":     0.00050,
    "kali_garang":       0.00055,
    "kali_mas":          0.00038,
    "kali_surabaya":     0.00042,
    "kali_tuntang":      0.00032,
    "kali_serang":       0.00030,
    "kali_pekalongan":   0.00045,
    "kali_sragi":        0.00040,
}

# Absolute SLR (m/yr) — Jakarta Strait / Java Sea
_ABSOLUTE_SLR_M_YR = 0.003   # 3mm/yr


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IntrusionResult:
    city_id:                    str
    date:                       str
    sii:                        float
    sii_category:               str         # SAFE|MODERATE|HIGH|CRITICAL
    salinity_proxy_ppt:         float
    intrusion_km:               float       # estimated upstream extent
    affected_groundwater_wells: int
    tide_level_m:               float
    river_discharge_cms:        float
    output_path:                str
    notes:                      list[str] = field(default_factory=list)


@dataclass
class TidalBackwaterResult:
    river_id:            str
    city_id:             str
    date:                str
    discharge_cms:       float
    tide_level_m:        float
    backwater_km:        float
    backwater_flag:      bool        # True if > 15 km
    manning_n:           float
    slope:               float
    output_path:         str
    notes:               list[str] = field(default_factory=list)


@dataclass
class CoastalFloodRisk:
    city_id:                  str
    date:                     str
    sii:                      float
    sii_category:             str
    backwater_km:             float
    effective_slr_mm_yr:      float
    compound_flood_risk:      str   # LOW|MODERATE|HIGH|EXTREME
    tide_level_m:             float
    river_discharge_cms:      float
    recommended_gate_operation: str
    output_path:              str
    notes:                    list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CoastalHydrology:
    """
    Tidal–river discharge interaction and saltwater intrusion index for 5 subsidence-prone
    coastal cities. Integrates with flood_early_warning_dag (backwater modifier) and
    reservoir_operations_dag (Jatiluhur freshwater flushing for Jakarta SII > 0.7).
    """

    WORKSPACE    = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    CONFIG_FILE  = WORKSPACE / "config" / "coastal_cities_config.json"
    TIDAL_DIR    = WORKSPACE / "data" / "tidal"
    SF_SIDECAR   = WORKSPACE / "output" / "streamflow" / "latest_forecast.json"
    OUTPUT_DIR   = WORKSPACE / "output" / "coastal"
    ACTIVE_RISK  = WORKSPACE / "output" / "coastal" / "active_coastal_risk.json"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_saltwater_intrusion(
        self,
        city_id:            str,
        dt:                 date | None = None,
        river_discharge_cms: float | None = None,
        tide_level_m:       float | None = None,
    ) -> IntrusionResult:
        """
        Compute Saltwater Intrusion Index (SII) and estimate intrusion extent.
        SII = (Q_tidal / Q_river) × (1 + RSL_anomaly_m)
        """
        if dt is None:
            dt = datetime.now(timezone.utc).date()

        city_l = city_id.lower()
        cfg    = self._get_cfg(city_l)
        notes: list[str] = []

        # Tidal parameters
        tidal_eff   = cfg.get("tidal_efficiency", 0.60)
        tidal_prism = cfg.get("tidal_prism_m3",  1_000_000.0)

        if tide_level_m is None:
            tide_level_m = self._load_tide(city_l, dt, notes)
        if river_discharge_cms is None:
            river_discharge_cms = self._load_discharge(city_l, dt, notes)

        # Q_tidal = tidal prism × efficiency / tidal half-period (6hr = 21600s)
        q_tidal = tidal_prism * tidal_eff * tide_level_m / 21600.0
        q_river = max(river_discharge_cms, 0.1)

        # Relative sea level anomaly
        subsidence_m = cfg.get("land_subsidence_mm_yr", 15) / 1000.0
        rsl_anomaly  = (subsidence_m + _ABSOLUTE_SLR_M_YR) * tide_level_m

        sii      = (q_tidal / q_river) * (1.0 + rsl_anomaly)
        sii      = round(sii, 4)
        sii_cat  = self._sii_category(sii)

        # Salinity proxy (simplified mixing): PSS ≈ 35 × SII / (1 + SII)
        salinity = round(35.0 * sii / (1.0 + sii), 2)

        # Intrusion extent: km ≈ 5 × SII^0.6
        intrusion_km = round(5.0 * (sii ** 0.6), 2)

        # Affected wells (from spatial config, proportional to intrusion)
        total_wells = cfg.get("groundwater_wells_count", 5000)
        affected    = int(total_wells * min(intrusion_km / 20.0, 1.0))

        # Prometheus
        self._emit_metric(city_l, sii)

        out_path = self.OUTPUT_DIR / f"intrusion_{city_l}_{dt.strftime('%Y%m%d')}.json"
        result = IntrusionResult(
            city_id                    = city_l,
            date                       = dt.isoformat(),
            sii                        = sii,
            sii_category               = sii_cat,
            salinity_proxy_ppt         = salinity,
            intrusion_km               = intrusion_km,
            affected_groundwater_wells = affected,
            tide_level_m               = round(tide_level_m, 3),
            river_discharge_cms        = round(river_discharge_cms, 2),
            output_path                = str(out_path),
            notes                      = notes,
        )
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        if sii_cat in ("HIGH","CRITICAL"):
            logger.warning(
                "SALTWATER INTRUSION | %-12s SII=%.3f (%s) intrusion=%.1fkm Q_river=%.0f",
                city_l, sii, sii_cat, intrusion_km, river_discharge_cms,
            )
        else:
            logger.info(
                "Intrusion | %-12s SII=%.3f (%s)", city_l, sii, sii_cat,
            )
        return result

    def compute_tidal_backwater(
        self,
        river_id:      str,
        dt:            date | None = None,
        discharge_cms: float | None = None,
    ) -> TidalBackwaterResult:
        """
        Compute tidal backwater extension (km).
        BW_km = k × (tide_h / slope) × (1 / sqrt(discharge))
        Flags if > 15 km (modifies flood stage in flood_early_warning_dag).
        """
        if dt is None:
            dt = datetime.now(timezone.utc).date()

        # Find city for this river
        city_l = self._river_to_city(river_id)
        cfg    = self._get_cfg(city_l)
        notes: list[str] = []

        manning_n = cfg.get("manning_n", {}).get(river_id, 0.041)
        slope     = _RIVER_SLOPE.get(river_id, 0.00040)
        tide_h    = self._load_tide(city_l, dt, notes)
        if discharge_cms is None:
            discharge_cms = self._load_discharge(city_l, dt, notes)
        q = max(discharge_cms, 1.0)

        # Backwater formula
        bw_km = manning_n * (tide_h / max(slope, 1e-6)) * (1.0 / math.sqrt(q))
        bw_km = round(bw_km, 2)
        flag  = bw_km > 15.0

        if flag:
            notes.append(f"Backwater > 15km ({bw_km:.1f}km) — flood stage modifier active")

        out_path = self.OUTPUT_DIR / f"backwater_{river_id}_{dt.strftime('%Y%m%d')}.json"
        result = TidalBackwaterResult(
            river_id       = river_id,
            city_id        = city_l,
            date           = dt.isoformat(),
            discharge_cms  = round(q, 2),
            tide_level_m   = round(tide_h, 3),
            backwater_km   = bw_km,
            backwater_flag = flag,
            manning_n      = manning_n,
            slope          = slope,
            output_path    = str(out_path),
            notes          = notes,
        )
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(
            "Backwater | %-22s bw=%.1fkm flag=%s Q=%.0f tide=%.2f",
            river_id, bw_km, flag, q, tide_h,
        )
        return result

    def get_coastal_flood_risk(
        self,
        city_id:    str,
        intrusion:  IntrusionResult,
        backwater:  TidalBackwaterResult,
    ) -> CoastalFloodRisk:
        """
        Compound coastal flood risk combining SII + backwater extension + effective SLR.
        """
        city_l   = city_id.lower()
        cfg      = self._get_cfg(city_l)
        notes: list[str] = []

        subsidence_mm_yr  = cfg.get("land_subsidence_mm_yr", 15)
        eff_slr_mm_yr     = subsidence_mm_yr + _ABSOLUTE_SLR_M_YR * 1000.0

        # Compound risk matrix
        risk = self._compound_risk(intrusion.sii_category, backwater.backwater_km, eff_slr_mm_yr)

        # Gate operation recommendation
        gate_rec = self._gate_recommendation(
            city_l, intrusion.sii, intrusion.sii_category, backwater.backwater_km,
        )

        dt = datetime.now(timezone.utc).date()
        out_path = self.OUTPUT_DIR / f"coastal_{city_l}_{dt.strftime('%Y%m%d')}.json"
        result = CoastalFloodRisk(
            city_id                  = city_l,
            date                     = dt.isoformat(),
            sii                      = intrusion.sii,
            sii_category             = intrusion.sii_category,
            backwater_km             = backwater.backwater_km,
            effective_slr_mm_yr      = round(eff_slr_mm_yr, 1),
            compound_flood_risk      = risk,
            tide_level_m             = intrusion.tide_level_m,
            river_discharge_cms      = intrusion.river_discharge_cms,
            recommended_gate_operation = gate_rec,
            output_path              = str(out_path),
            notes                    = notes,
        )
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        if risk in ("HIGH","EXTREME"):
            logger.warning(
                "COASTAL FLOOD RISK | %-12s risk=%s SII=%s BW=%.1fkm eff_SLR=%.1fmm/yr",
                city_l, risk, intrusion.sii_category, backwater.backwater_km, eff_slr_mm_yr,
            )
        return result

    def update_active_coastal_risk(
        self,
        risks:     list[CoastalFloodRisk],
        timestamp: datetime,
    ) -> None:
        """Write rolling active_coastal_risk.json sidecar for GEOSPATIAL + VISUALIA."""
        risk_order = {"EXTREME":3,"HIGH":2,"MODERATE":1,"LOW":0}
        cities = {
            r.city_id: {
                "compound_flood_risk": r.compound_flood_risk,
                "sii":                 r.sii,
                "sii_category":        r.sii_category,
                "backwater_km":        r.backwater_km,
                "effective_slr_mm_yr": r.effective_slr_mm_yr,
                "tide_level_m":        r.tide_level_m,
                "river_discharge_cms": r.river_discharge_cms,
                "recommended_gate":    r.recommended_gate_operation,
                "date":                r.date,
            }
            for r in risks
        }
        highest = max(
            (cities[c]["compound_flood_risk"] for c in cities),
            key=lambda s: risk_order.get(s, 0),
            default="LOW",
        )
        sidecar = {
            "generated_at":    timestamp.isoformat(),
            "city_count":      len(cities),
            "highest_risk":    highest,
            "high_or_extreme": sum(1 for c in cities if risk_order.get(cities[c]["compound_flood_risk"],0)>=2),
            "cities":          cities,
        }
        with open(self.ACTIVE_RISK, "w") as f:
            json.dump(sidecar, f, indent=2)
        logger.info(
            "active_coastal_risk.json updated | %d cities | highest=%s",
            len(cities), highest,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sii_category(sii: float) -> str:
        for cat, thr in _SII_CATEGORIES:
            if sii >= thr:
                return cat
        return "SAFE"

    @staticmethod
    def _compound_risk(
        sii_cat:     str,
        backwater_km: float,
        eff_slr_mm_yr: float,
    ) -> str:
        score = 0
        score += {"CRITICAL":3,"HIGH":2,"MODERATE":1,"SAFE":0}.get(sii_cat, 0)
        if backwater_km > 25:   score += 2
        elif backwater_km > 15: score += 1
        if eff_slr_mm_yr > 25:  score += 1
        if   score >= 5: return "EXTREME"
        elif score >= 3: return "HIGH"
        elif score >= 1: return "MODERATE"
        return "LOW"

    @staticmethod
    def _gate_recommendation(
        city_id: str,
        sii:     float,
        sii_cat: str,
        bw_km:   float,
    ) -> str:
        if city_id == "jakarta" and sii > 0.7:
            return (
                "Prioritise Jatiluhur freshwater flushing release "
                f"(SII={sii:.2f} > 0.7 threshold). "
                "Close tidal gates on Ciliwung estuary. Alert PDAM Jakarta."
            )
        if sii_cat == "CRITICAL":
            return "Close all tidal exclusion gates. Emergency freshwater release. Alert BPBD."
        if sii_cat == "HIGH" or bw_km > 15:
            return "Reduce gate openings to 50%. Monitor tide gauge hourly. Prepare pumping stations."
        return "Normal gate operations. Continue routine tide monitoring."

    def _load_tide(self, city_id: str, dt: date, notes: list[str]) -> float:
        """Load tidal level from data/tidal/ or synthetic."""
        cfg      = self._get_cfg(city_id)
        gauge    = cfg.get("sea_level_gauge", "")
        date_str = dt.strftime("%Y%m%d")
        path     = self.TIDAL_DIR / f"{gauge}_{date_str}.json"
        if path.exists():
            try:
                with open(path) as f:
                    return float(json.load(f).get("max_tide_m", 0.7))
            except Exception:
                pass
        notes.append(f"Tidal gauge data absent for {city_id}; using synthetic")
        rng = random.Random(hash(f"tide{city_id}{date_str}"))
        return round(rng.uniform(0.4, 1.2), 3)

    def _load_discharge(self, city_id: str, dt: date, notes: list[str]) -> float:
        """Load river discharge from streamflow sidecar or synthetic."""
        if self.SF_SIDECAR.exists():
            try:
                with open(self.SF_SIDECAR) as f:
                    data = json.load(f)
                q = data.get("cities", {}).get(city_id, {}).get("discharge_cms")
                if q:
                    return float(q)
            except Exception:
                pass
        notes.append(f"Streamflow sidecar absent for {city_id}; using synthetic")
        rng = random.Random(hash(f"q{city_id}{dt.strftime('%Y%m%d')}"))
        return round(rng.uniform(30.0, 250.0), 2)

    def _river_to_city(self, river_id: str) -> str:
        for cid, cfg in self._config.items():
            if river_id in cfg.get("rivers", []):
                return cid
        return "jakarta"

    def _get_cfg(self, city_id: str) -> dict:
        cfg = self._config.get(city_id.lower())
        if cfg is None:
            logger.warning("Unknown city_id '%s'; using defaults", city_id)
            return {"land_subsidence_mm_yr": 15, "tidal_efficiency": 0.60,
                    "tidal_prism_m3": 1_000_000, "rivers": [], "groundwater_wells_count": 5000}
        return cfg

    def _load_config(self) -> dict:
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _emit_metric(city_id: str, sii: float) -> None:
        try:
            from src.hydrology.metrics import SALTWATER_INTRUSION_INDEX
            SALTWATER_INTRUSION_INDEX.labels(city_id=city_id).set(sii)
        except Exception as exc:
            logger.debug("SALTWATER_INTRUSION_INDEX emit error: %s", exc)
