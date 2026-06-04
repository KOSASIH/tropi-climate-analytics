"""
watershed_water_balance.py — Sprint 7 G1
WatershedWaterBalance: Monthly water balance closure for 20 DAS Strategis Nasional.

Components:
  P  = QPE monthly accumulation (workspace/data/qpe_monthly_{YYYYMM}.json)
  ET = MODIS ET product (workspace/data/modis_et_{YYYYMM}.json) or
       Penman-Monteith estimate fallback
  Q  = streamflow_forecast.py monthly integral
       (sum of daily peak_cms × 86400 / catchment_area_km2)
  ΔS = SMAP root-zone ΔθV + GRACE-FO TWS anomaly (grace_aquifer_depletion output)

Balance: residual = P - ET - Q - ΔS
  |residual / P| > 0.15 → status='UNCLOSED', else 'CLOSED'

Output: workspace/output/water_balance/balance_{watershed_id}_{YYYYMM}.json
Prometheus: tropi_water_balance_residual_fraction{watershed_id} Gauge
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAS catalogue — catchment areas (km²) and Penman-Monteith ET params
# ---------------------------------------------------------------------------
_DAS_PARAMS: dict[str, dict[str, Any]] = {
    "DAS-CI":  {"name": "Ciliwung",       "area_km2": 447.0,  "pm_et_mm": 112.0},
    "DAS-BR":  {"name": "Brantas",        "area_km2": 12000.0,"pm_et_mm": 95.0},
    "DAS-SL":  {"name": "Solo",           "area_km2": 16100.0,"pm_et_mm": 98.0},
    "DAS-MK":  {"name": "Musi-Komering",  "area_km2": 60700.0,"pm_et_mm": 108.0},
    "DAS-KP":  {"name": "Kapuas",         "area_km2": 97800.0,"pm_et_mm": 115.0},
    "DAS-ML":  {"name": "Mahakam",        "area_km2": 77100.0,"pm_et_mm": 110.0},
    "DAS-SN":  {"name": "Sungai Negara",  "area_km2": 14200.0,"pm_et_mm": 105.0},
    "DAS-MN":  {"name": "Mamberamo",      "area_km2": 78000.0,"pm_et_mm": 125.0},
    "DAS-PW":  {"name": "Progo-Opak",     "area_km2": 4540.0, "pm_et_mm": 100.0},
    "DAS-JR":  {"name": "Jratunseluna",   "area_km2": 9470.0, "pm_et_mm": 102.0},
    "DAS-SR":  {"name": "Serayu",         "area_km2": 3680.0, "pm_et_mm": 103.0},
    "DAS-CR":  {"name": "Citarum",        "area_km2": 6614.0, "pm_et_mm": 105.0},
    "DAS-CM":  {"name": "Cimanuk",        "area_km2": 3600.0, "pm_et_mm": 104.0},
    "DAS-CJ":  {"name": "Cisadane",       "area_km2": 1480.0, "pm_et_mm": 106.0},
    "DAS-BI":  {"name": "Barito",         "area_km2": 60400.0,"pm_et_mm": 112.0},
    "DAS-WK":  {"name": "Walanae-Cenrana","area_km2": 11400.0,"pm_et_mm": 96.0},
    "DAS-PO":  {"name": "Poso",           "area_km2": 3890.0, "pm_et_mm": 105.0},
    "DAS-TA":  {"name": "Tondano",        "area_km2": 1340.0, "pm_et_mm": 108.0},
    "DAS-TM":  {"name": "Tabalong",       "area_km2": 7810.0, "pm_et_mm": 107.0},
    "DAS-DG":  {"name": "Digul",          "area_km2": 35900.0,"pm_et_mm": 128.0},
}

_RESIDUAL_THRESHOLD = 0.15  # 15 % of P → UNCLOSED


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WaterBalanceResult:
    watershed_id:     str
    watershed_name:   str
    month:            str           # YYYY-MM
    P_mm:             float         # precipitation
    ET_mm:            float         # evapotranspiration
    Q_mm:             float         # runoff (streamflow integral → mm)
    delta_S_mm:       float         # storage change
    residual_mm:      float         # P - ET - Q - ΔS
    residual_fraction:float         # |residual| / P
    status:           str           # "CLOSED" | "UNCLOSED"
    et_source:        str           # "modis" | "penman_monteith"
    ds_source:        str           # "smap+grace" | "smap_only" | "zero"
    output_path:      str
    warnings:         list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WatershedWaterBalance:
    """
    Monthly water balance computer for 20 DAS Strategis Nasional.

    All components converted to mm over the watershed area before balance.
    """

    WORKSPACE  = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    DATA_DIR   = WORKSPACE / "data"
    OUTPUT_DIR = WORKSPACE / "output" / "water_balance"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        watershed_id: str,
        month: date | None = None,
    ) -> WaterBalanceResult:
        """
        Compute monthly water balance for *watershed_id* in *month*.
        Returns WaterBalanceResult with closure status.
        """
        if month is None:
            month = datetime.now(timezone.utc).date().replace(day=1)

        if watershed_id not in _DAS_PARAMS:
            raise ValueError(
                f"Unknown watershed_id '{watershed_id}'. "
                f"Supported: {list(_DAS_PARAMS.keys())}"
            )

        das       = _DAS_PARAMS[watershed_id]
        month_str = month.strftime("%Y%m")
        warnings: list[str] = []

        # --- P: QPE monthly accumulation ---
        P_mm, p_warn = self._load_qpe_monthly(watershed_id, month_str)
        warnings.extend(p_warn)

        # --- ET: MODIS or Penman-Monteith ---
        ET_mm, et_source, et_warn = self._load_et(watershed_id, month_str, das)
        warnings.extend(et_warn)

        # --- Q: streamflow monthly integral → mm ---
        Q_mm, q_warn = self._compute_streamflow_mm(watershed_id, das["area_km2"])
        warnings.extend(q_warn)

        # --- ΔS: SMAP + GRACE-FO ---
        dS_mm, ds_source, ds_warn = self._compute_delta_storage(watershed_id, month_str)
        warnings.extend(ds_warn)

        # --- Balance ---
        residual = P_mm - ET_mm - Q_mm - dS_mm
        frac     = abs(residual / P_mm) if P_mm > 0 else 0.0
        status   = "UNCLOSED" if frac > _RESIDUAL_THRESHOLD else "CLOSED"

        # --- Prometheus ---
        self._emit_metric(watershed_id, frac)

        # --- Output ---
        out_path = self.OUTPUT_DIR / f"balance_{watershed_id}_{month_str}.json"
        result   = WaterBalanceResult(
            watershed_id      = watershed_id,
            watershed_name    = das["name"],
            month             = month.strftime("%Y-%m"),
            P_mm              = round(P_mm, 2),
            ET_mm             = round(ET_mm, 2),
            Q_mm              = round(Q_mm, 2),
            delta_S_mm        = round(dS_mm, 2),
            residual_mm       = round(residual, 2),
            residual_fraction = round(frac, 4),
            status            = status,
            et_source         = et_source,
            ds_source         = ds_source,
            output_path       = str(out_path),
            warnings          = warnings,
        )

        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(
            "Water balance | %s %s P=%.1f ET=%.1f Q=%.1f ΔS=%.1f res=%.1f (%s)",
            watershed_id, month.strftime("%Y-%m"),
            P_mm, ET_mm, Q_mm, dS_mm, residual, status,
        )
        return result

    # ------------------------------------------------------------------
    # Component loaders
    # ------------------------------------------------------------------

    def _load_qpe_monthly(
        self, watershed_id: str, month_str: str
    ) -> tuple[float, list[str]]:
        warnings: list[str] = []
        path = self.DATA_DIR / f"qpe_monthly_{month_str}.json"
        if not path.exists():
            warnings.append(f"QPE monthly file absent ({path}); using 0 mm for P.")
            return 0.0, warnings
        with open(path) as f:
            data = json.load(f)
        # Expect { "<DAS_ID>": {"accumulation_mm": <float>}, ... }
        # or top-level accumulation_mm if single-watershed file
        entry = data.get(watershed_id, data)
        P_mm  = float(entry.get("accumulation_mm", 0.0) or 0.0)
        if P_mm == 0.0:
            warnings.append(f"QPE accumulation is 0 for {watershed_id} {month_str}")
        return P_mm, warnings

    def _load_et(
        self,
        watershed_id: str,
        month_str:    str,
        das:          dict[str, Any],
    ) -> tuple[float, str, list[str]]:
        warnings: list[str] = []
        path = self.DATA_DIR / f"modis_et_{month_str}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                entry = data.get(watershed_id, data)
                et_mm = float(entry.get("et_mm", 0.0) or 0.0)
                if et_mm > 0:
                    return et_mm, "modis", warnings
                warnings.append(f"MODIS ET = 0 for {watershed_id}; falling back to Penman-Monteith")
            except Exception as exc:
                warnings.append(f"MODIS ET parse error ({exc}); falling back to Penman-Monteith")
        else:
            warnings.append(f"MODIS ET file absent ({path}); using Penman-Monteith estimate")
        # Penman-Monteith climatological estimate (mm/month)
        return das["pm_et_mm"], "penman_monteith", warnings

    def _compute_streamflow_mm(
        self,
        watershed_id: str,
        area_km2:     float,
    ) -> tuple[float, list[str]]:
        """
        Monthly streamflow integral from latest forecast outputs.
        Q_mm = Σ(peak_cms_daily × 86400 s) / (area_km2 × 1e6 m²) × 1000 mm/m
             = Σ(peak_cms × 86400) / (area_km2 × 1e3)
        Reads available forecast JSON files; falls back to 0 if none.
        """
        warnings: list[str] = []
        forecast_dir = self.WORKSPACE / "output" / "streamflow"

        # Map DAS watershed to river
        das_to_river = {
            "DAS-CI": "ciliwung", "DAS-BR": "brantas", "DAS-SL": "solo",
            "DAS-PW": "ciliwung", "DAS-JR": "brantas", "DAS-SR": "brantas",
            "DAS-CR": "ciliwung", "DAS-CM": "ciliwung", "DAS-CJ": "ciliwung",
        }
        river_id = das_to_river.get(watershed_id)
        if not river_id or not forecast_dir.exists():
            warnings.append(
                f"No streamflow river mapping for {watershed_id}; Q set to 0 mm"
            )
            return 0.0, warnings

        files = sorted(forecast_dir.glob(f"forecast_{river_id}_*.json"))
        if not files:
            warnings.append(f"No streamflow forecast files for {river_id}; Q set to 0 mm")
            return 0.0, warnings

        # Use most recent file
        latest = files[-1]
        try:
            with open(latest) as f:
                data = json.load(f)
            # Sum peak_cms across horizons as daily representative value
            peak_cms_sum = sum(
                h.get("peak_cms", 0.0)
                for h in data.get("horizons", [])
            )
            avg_peak = peak_cms_sum / max(len(data.get("horizons", [1])), 1)
            # Monthly integral: avg_peak × 30 days × 86400 s / day
            Q_m3 = avg_peak * 30 * 86400
            Q_mm = Q_m3 / (area_km2 * 1e6) * 1000
            return round(Q_mm, 3), warnings
        except Exception as exc:
            warnings.append(f"Streamflow integral error ({exc}); Q set to 0 mm")
            return 0.0, warnings

    def _compute_delta_storage(
        self,
        watershed_id: str,
        month_str:    str,
    ) -> tuple[float, str, list[str]]:
        """
        ΔS = SMAP root-zone ΔθV (mm) + GRACE-FO GWS anomaly (mm).
        Falls back to SMAP only or zero.
        """
        warnings: list[str] = []
        dS_smap  = 0.0
        dS_grace = 0.0
        source   = "zero"

        # SMAP: read current and previous month θ files
        smap_curr = self.DATA_DIR / "smap_latest.json"
        if smap_curr.exists():
            try:
                with open(smap_curr) as f:
                    smap = json.load(f)
                entry = smap.get(watershed_id, {}) if isinstance(smap, dict) else {}
                theta = float(entry.get("theta_root_zone", 0.0) or 0.0)
                theta_prev = float(entry.get("theta_prev_month", theta) or theta)
                depth_mm   = 1000.0  # 1 m root zone depth
                dS_smap    = (theta - theta_prev) * depth_mm
                source     = "smap_only"
            except Exception as exc:
                warnings.append(f"SMAP delta storage error ({exc})")

        # GRACE-FO: from grace_aquifer_depletion output
        grace_path = self.WORKSPACE / "output" / "aquifer" / f"grace_depletion_{month_str}.json"
        if grace_path.exists():
            try:
                with open(grace_path) as f:
                    grace = json.load(f)
                # Sum GWS anomaly across regions covering this DAS
                regions  = grace.get("regions", [])
                matching = [r for r in regions if watershed_id in r.get("das_ids", [])]
                if matching:
                    dS_grace = sum(r.get("delta_gws_mm", 0.0) for r in matching) / max(len(matching), 1)
                    source   = "smap+grace"
                else:
                    # Use basin-mean GWS if no specific match
                    dS_grace = float(grace.get("mean_delta_gws_mm", 0.0) or 0.0)
                    if dS_grace != 0.0:
                        source = "smap+grace"
            except Exception as exc:
                warnings.append(f"GRACE-FO storage error ({exc})")

        dS_total = dS_smap + dS_grace
        return dS_total, source, warnings

    @staticmethod
    def _emit_metric(watershed_id: str, residual_fraction: float) -> None:
        try:
            from src.hydrology.metrics import WATER_BALANCE_RESIDUAL
            WATER_BALANCE_RESIDUAL.labels(watershed_id=watershed_id).set(residual_fraction)
        except Exception as exc:
            logger.debug("Prometheus metrics unavailable: %s", exc)
