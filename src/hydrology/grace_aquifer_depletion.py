"""
grace_aquifer_depletion.py — Sprint 7 G5
GRACEAquiferDepletion: Monthly groundwater storage anomaly from GRACE-FO
mascon data, decomposed into GWS = TWS - SMS - SWS.

TWS decomposition:
  ΔGWS = ΔTWS - ΔSMS - ΔSWS
    ΔTWS: GRACE-FO mascon (workspace/data/grace/grace_{YYYYMM}.json, stub)
    ΔSMS: SMAP root-zone soil moisture change
    ΔSWS: surface water storage change (estimate from QPE - runoff)

Thresholds (mm/month):
  ΔGWS > -5     = NORMAL
  -5  to -10    = WATCH
  -10 to -15    = WARNING
  < -15         = EMERGENCY

Output: workspace/output/aquifer/grace_depletion_{YYYYMM}.json
Feeds into: watershed_water_balance.py (ΔS component)

Prometheus:
  tropi_grace_gws_anomaly_mm{region_id}  Gauge
  tropi_aquifer_emergency_total          Counter
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
# Region catalogue — GRACE mascon regions covering Indonesian aquifer systems
# ---------------------------------------------------------------------------
_REGIONS: list[dict[str, Any]] = [
    {
        "id":      "GWR-JW",
        "name":    "Java",
        "das_ids": ["DAS-CI", "DAS-BR", "DAS-SL", "DAS-PW", "DAS-JR",
                    "DAS-SR", "DAS-CR", "DAS-CM", "DAS-CJ"],
        "area_km2": 128297.0,
        "theta_clim": 0.30,
    },
    {
        "id":      "GWR-SM",
        "name":    "Sumatra",
        "das_ids": ["DAS-MK"],
        "area_km2": 473481.0,
        "theta_clim": 0.33,
    },
    {
        "id":      "GWR-KL",
        "name":    "Kalimantan",
        "das_ids": ["DAS-KP", "DAS-ML", "DAS-SN", "DAS-BI", "DAS-TM"],
        "area_km2": 748168.0,
        "theta_clim": 0.35,
    },
    {
        "id":      "GWR-SL",
        "name":    "Sulawesi",
        "das_ids": ["DAS-WK", "DAS-PO", "DAS-TA"],
        "area_km2": 186216.0,
        "theta_clim": 0.31,
    },
    {
        "id":      "GWR-PP",
        "name":    "Papua",
        "das_ids": ["DAS-MN", "DAS-DG"],
        "area_km2": 421981.0,
        "theta_clim": 0.38,
    },
]

# Depletion thresholds (mm/month) — lower bound is more negative
_THRESHOLDS = [
    (-15.0, "EMERGENCY", 3),
    (-10.0, "WARNING",   2),
    (-5.0,  "WATCH",     1),
    (float("-inf"), "NORMAL", 0),
]

# Root-zone depth (m) for SMAP θ → mm conversion
_SOIL_DEPTH_M = 1.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegionGWSResult:
    region_id:       str
    region_name:     str
    das_ids:         list[str]
    delta_tws_mm:    float       # GRACE-FO ΔTWS
    delta_sms_mm:    float       # SMAP ΔSMS
    delta_sws_mm:    float       # QPE-runoff ΔSWS
    delta_gws_mm:    float       # = ΔTWS - ΔSMS - ΔSWS
    depletion_label: str         # NORMAL/WATCH/WARNING/EMERGENCY
    depletion_class: int         # 0–3
    tws_source:      str         # "grace" | "climatological"


@dataclass
class AquiferDepletionResult:
    month:             str              # YYYY-MM
    run_ts:            str
    regions:           list[RegionGWSResult]
    mean_delta_gws_mm: float            # basin-mean across all regions
    emergency_count:   int
    output_path:       str
    status:            str              # "ok" | "degraded"
    notes:             list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GRACEAquiferDepletion:
    """
    Monthly groundwater storage anomaly estimator from GRACE-FO mascon data.
    Falls back to climatological TWS trend when GRACE is unavailable.
    """

    WORKSPACE   = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    DATA_DIR    = WORKSPACE / "data"
    GRACE_DIR   = DATA_DIR / "grace"
    SMAP_PATH   = DATA_DIR / "smap_latest.json"
    OUTPUT_DIR  = WORKSPACE / "output" / "aquifer"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_monthly(self, month: date | None = None) -> AquiferDepletionResult:
        """
        Compute GRACE-FO groundwater storage anomaly for all regions in *month*.
        Returns AquiferDepletionResult with per-region GWS scores.
        """
        if month is None:
            month = datetime.now(timezone.utc).date().replace(day=1)

        month_str = month.strftime("%Y%m")
        run_ts    = datetime.now(timezone.utc).isoformat()
        notes:    list[str] = []

        # --- Load GRACE-FO ---
        grace_data, grace_source, grace_notes = self._load_grace(month_str)
        notes.extend(grace_notes)

        # --- Load SMAP ---
        smap_data, smap_notes = self._load_smap()
        notes.extend(smap_notes)

        # --- Load QPE monthly (for ΔSWS) ---
        qpe_data, qpe_notes = self._load_qpe_monthly(month_str)
        notes.extend(qpe_notes)

        # --- Compute per-region GWS ---
        region_results: list[RegionGWSResult] = []
        emergency_count = 0

        for region in _REGIONS:
            result = self._compute_region(
                region, grace_data, grace_source, smap_data, qpe_data
            )
            region_results.append(result)
            if result.depletion_class == 3:
                emergency_count += 1

            # Emit Prometheus
            self._emit_region_metric(result)

        mean_gws = (
            sum(r.delta_gws_mm for r in region_results) / len(region_results)
            if region_results else 0.0
        )

        if emergency_count > 0:
            try:
                from src.hydrology.metrics import AQUIFER_EMERGENCY
                AQUIFER_EMERGENCY.inc(emergency_count)
            except Exception as exc:
                logger.debug("Prometheus AQUIFER_EMERGENCY unavailable: %s", exc)

        status   = "degraded" if grace_source == "climatological" else "ok"

        # --- Write output ---
        out_path = self.OUTPUT_DIR / f"grace_depletion_{month_str}.json"
        result_obj = AquiferDepletionResult(
            month             = month.strftime("%Y-%m"),
            run_ts            = run_ts,
            regions           = region_results,
            mean_delta_gws_mm = round(mean_gws, 3),
            emergency_count   = emergency_count,
            output_path       = str(out_path),
            status            = status,
            notes             = notes,
        )

        with open(out_path, "w") as f:
            json.dump(asdict(result_obj), f, indent=2)

        logger.info(
            "GRACE aquifer depletion | month=%s status=%s mean_ΔGWS=%.2f mm emergency=%d",
            month.strftime("%Y-%m"), status, mean_gws, emergency_count,
        )
        return result_obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_grace(
        self, month_str: str
    ) -> tuple[dict[str, Any], str, list[str]]:
        notes: list[str] = []
        path  = self.GRACE_DIR / f"grace_{month_str}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data, "grace", notes
            except Exception as exc:
                notes.append(f"GRACE file parse error ({exc}); using climatological fallback")
        else:
            notes.append(
                f"GRACE-FO mascon file absent ({path}); "
                "using climatological TWS trend (-2 mm/month Indonesia mean)."
            )
        return {}, "climatological", notes

    def _load_smap(self) -> tuple[dict[str, Any], list[str]]:
        notes: list[str] = []
        if not self.SMAP_PATH.exists():
            notes.append("SMAP file absent; ΔSMS set to 0 mm")
            return {}, notes
        try:
            with open(self.SMAP_PATH) as f:
                return json.load(f), notes
        except Exception as exc:
            notes.append(f"SMAP parse error ({exc}); ΔSMS set to 0 mm")
            return {}, notes

    def _load_qpe_monthly(
        self, month_str: str
    ) -> tuple[dict[str, Any], list[str]]:
        notes: list[str] = []
        path  = self.DATA_DIR / f"qpe_monthly_{month_str}.json"
        if not path.exists():
            notes.append(f"QPE monthly file absent ({path}); ΔSWS set to 0 mm")
            return {}, notes
        try:
            with open(path) as f:
                return json.load(f), notes
        except Exception as exc:
            notes.append(f"QPE monthly parse error ({exc}); ΔSWS set to 0 mm")
            return {}, notes

    def _compute_region(
        self,
        region:      dict[str, Any],
        grace_data:  dict[str, Any],
        grace_source:str,
        smap_data:   dict[str, Any],
        qpe_data:    dict[str, Any],
    ) -> RegionGWSResult:
        rid = region["id"]

        # ΔTWS from GRACE or climatological fallback (-2 mm/month)
        if grace_source == "grace":
            entry    = grace_data.get(rid, {})
            delta_tws = float(entry.get("delta_tws_mm", -2.0) if isinstance(entry, dict) else -2.0)
        else:
            delta_tws = -2.0  # Indonesian mean climatological trend

        # ΔSMS from SMAP (mean over DAS in region)
        delta_sms = self._compute_delta_sms(region, smap_data)

        # ΔSWS = QPE - runoff approximation (simplified: 30 % of QPE retained as SWS change)
        delta_sws = self._compute_delta_sws(region, qpe_data)

        # GWS anomaly
        delta_gws = delta_tws - delta_sms - delta_sws
        label, cls = self._classify(delta_gws)

        return RegionGWSResult(
            region_id       = rid,
            region_name     = region["name"],
            das_ids         = region["das_ids"],
            delta_tws_mm    = round(delta_tws, 3),
            delta_sms_mm    = round(delta_sms, 3),
            delta_sws_mm    = round(delta_sws, 3),
            delta_gws_mm    = round(delta_gws, 3),
            depletion_label = label,
            depletion_class = cls,
            tws_source      = grace_source,
        )

    @staticmethod
    def _compute_delta_sms(
        region:    dict[str, Any],
        smap_data: dict[str, Any],
    ) -> float:
        """Mean ΔSMS (mm) across DAS in region from SMAP data."""
        if not smap_data or not isinstance(smap_data, dict):
            return 0.0
        deltas = []
        for das_id in region["das_ids"]:
            entry = smap_data.get(das_id, {})
            if not isinstance(entry, dict):
                continue
            theta     = float(entry.get("theta_root_zone", 0.0) or 0.0)
            theta_prev= float(entry.get("theta_prev_month", theta) or theta)
            deltas.append((theta - theta_prev) * _SOIL_DEPTH_M * 1000.0)
        return sum(deltas) / len(deltas) if deltas else 0.0

    @staticmethod
    def _compute_delta_sws(
        region:   dict[str, Any],
        qpe_data: dict[str, Any],
    ) -> float:
        """
        Estimate ΔSWS (mm) as 30% of QPE monthly accumulation
        (simplification: runoff ≈ 70 % of precip, SWS change ≈ 30 %).
        """
        if not qpe_data or not isinstance(qpe_data, dict):
            return 0.0
        vals = []
        for das_id in region["das_ids"]:
            entry  = qpe_data.get(das_id, qpe_data)
            P_mm   = float(entry.get("accumulation_mm", 0.0) or 0.0) if isinstance(entry, dict) else 0.0
            vals.append(P_mm * 0.30)
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _classify(delta_gws_mm: float) -> tuple[str, int]:
        for threshold, label, cls in _THRESHOLDS:
            if delta_gws_mm <= threshold:
                return label, cls
        return "NORMAL", 0

    @staticmethod
    def _emit_region_metric(result: "RegionGWSResult") -> None:
        try:
            from src.hydrology.metrics import GRACE_GWS_ANOMALY
            GRACE_GWS_ANOMALY.labels(region_id=result.region_id).set(result.delta_gws_mm)
        except Exception as exc:
            logger.debug("Prometheus metrics unavailable: %s", exc)
