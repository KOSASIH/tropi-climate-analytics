"""
drought_risk_monitor.py — Sprint 6 E3
DroughtRiskMonitor: SMAP root-zone soil moisture drought risk scoring for
20 DAS Strategis Nasional.

Input:  workspace/data/smap_latest.json  (stub; falls back to climatological mean)
Output: workspace/output/drought/drought_risk_{YYYYMM}.json

Risk classes:
  0 = NORMAL    (SWD < 20%)
  1 = WATCH     (20% ≤ SWD < 40%)
  2 = WARNING   (40% ≤ SWD < 60%)
  3 = EMERGENCY (SWD ≥ 60%)

SWD = (θ_fc − θ_obs) / (θ_fc − θ_wp) × 100  [%]
  θ_obs   — SMAP root-zone soil moisture [m³/m³]
  θ_fc    — field capacity [m³/m³]  (per DAS texture)
  θ_wp    — wilting point [m³/m³]   (per DAS texture)

Prometheus:
  tropi_drought_risk_class{watershed_id}     Gauge (0-3)
  tropi_drought_watersheds_warning_total     Gauge
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
# DAS catalogue — 20 DAS Strategis Nasional
# Soil texture parameters: θ_fc (field capacity), θ_wp (wilting point)  [m³/m³]
# Climatological mean θ_obs used as fallback when SMAP is absent
# ---------------------------------------------------------------------------
_DAS_CATALOGUE: list[dict[str, Any]] = [
    {"id": "DAS-CI",  "name": "Ciliwung",      "theta_fc": 0.38, "theta_wp": 0.14, "theta_clim": 0.30},
    {"id": "DAS-BR",  "name": "Brantas",        "theta_fc": 0.40, "theta_wp": 0.16, "theta_clim": 0.32},
    {"id": "DAS-SL",  "name": "Solo",           "theta_fc": 0.39, "theta_wp": 0.15, "theta_clim": 0.31},
    {"id": "DAS-MK",  "name": "Musi-Komering",  "theta_fc": 0.42, "theta_wp": 0.17, "theta_clim": 0.33},
    {"id": "DAS-KP",  "name": "Kapuas",         "theta_fc": 0.44, "theta_wp": 0.18, "theta_clim": 0.36},
    {"id": "DAS-ML",  "name": "Mahakam",        "theta_fc": 0.43, "theta_wp": 0.17, "theta_clim": 0.35},
    {"id": "DAS-SN",  "name": "Sungai Negara",  "theta_fc": 0.41, "theta_wp": 0.16, "theta_clim": 0.33},
    {"id": "DAS-MN",  "name": "Mamberamo",      "theta_fc": 0.45, "theta_wp": 0.19, "theta_clim": 0.38},
    {"id": "DAS-PW",  "name": "Progo-Opak",     "theta_fc": 0.37, "theta_wp": 0.14, "theta_clim": 0.29},
    {"id": "DAS-JR",  "name": "Jratunseluna",   "theta_fc": 0.38, "theta_wp": 0.15, "theta_clim": 0.30},
    {"id": "DAS-SR",  "name": "Serayu",         "theta_fc": 0.39, "theta_wp": 0.15, "theta_clim": 0.31},
    {"id": "DAS-CR",  "name": "Citarum",        "theta_fc": 0.40, "theta_wp": 0.16, "theta_clim": 0.32},
    {"id": "DAS-CM",  "name": "Cimanuk",        "theta_fc": 0.37, "theta_wp": 0.14, "theta_clim": 0.29},
    {"id": "DAS-CJ",  "name": "Cisadane",       "theta_fc": 0.38, "theta_wp": 0.15, "theta_clim": 0.30},
    {"id": "DAS-BI",  "name": "Barito",         "theta_fc": 0.43, "theta_wp": 0.18, "theta_clim": 0.35},
    {"id": "DAS-WK",  "name": "Walanae-Cenrana","theta_fc": 0.36, "theta_wp": 0.13, "theta_clim": 0.28},
    {"id": "DAS-PO",  "name": "Poso",           "theta_fc": 0.41, "theta_wp": 0.16, "theta_clim": 0.33},
    {"id": "DAS-TA",  "name": "Tondano",        "theta_fc": 0.39, "theta_wp": 0.15, "theta_clim": 0.31},
    {"id": "DAS-TM",  "name": "Tabalong",       "theta_fc": 0.42, "theta_wp": 0.17, "theta_clim": 0.34},
    {"id": "DAS-DG",  "name": "Digul",          "theta_fc": 0.46, "theta_wp": 0.20, "theta_clim": 0.39},
]

# Risk class boundaries (SWD %)
_RISK_THRESHOLDS = [
    (60.0, 3, "EMERGENCY"),
    (40.0, 2, "WARNING"),
    (20.0, 1, "WATCH"),
    (0.0,  0, "NORMAL"),
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WatershedDroughtScore:
    watershed_id:   str
    watershed_name: str
    theta_obs:      float    # SMAP root-zone θ [m³/m³]
    swd_pct:        float    # soil water deficit [%]
    risk_class:     int      # 0–3
    risk_label:     str      # NORMAL/WATCH/WARNING/EMERGENCY
    source:         str      # "smap" | "climatological"


@dataclass
class DroughtRiskResult:
    assessment_date:       str          # ISO date
    run_ts:                str          # ISO-8601 UTC
    watersheds:            list[WatershedDroughtScore]
    warning_count:         int          # risk_class >= 1
    emergency_count:       int          # risk_class == 3
    imbalanced_watersheds: list[str]    # ≥WARNING for downstream alert
    output_path:           str
    status:                str          # "ok" | "degraded"
    notes:                 list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DroughtRiskMonitor:
    """
    SMAP-based drought risk monitor for 20 DAS Strategis Nasional.

    Soil Water Deficit (SWD) is derived from SMAP root-zone soil moisture
    relative to field capacity and wilting point for each DAS texture class.
    Falls back to climatological mean when SMAP data is absent.
    """

    WORKSPACE   = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    DATA_DIR    = WORKSPACE / "data"
    OUTPUT_DIR  = WORKSPACE / "output" / "drought"
    SMAP_PATH   = DATA_DIR / "smap_latest.json"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, assessment_date: date | None = None) -> DroughtRiskResult:
        """
        Assess drought risk for all 20 DAS on *assessment_date*.
        Returns DroughtRiskResult with per-watershed scores and aggregates.
        """
        if assessment_date is None:
            assessment_date = datetime.now(timezone.utc).date()

        run_ts = datetime.now(timezone.utc).isoformat()
        notes:  list[str] = []

        # --- load SMAP (fallback to climatological mean) ---
        smap_data, smap_notes, smap_source = self._load_smap()
        notes.extend(smap_notes)

        # --- score each DAS ---
        scores: list[WatershedDroughtScore] = []
        for das in _DAS_CATALOGUE:
            theta_obs = self._get_theta(das, smap_data, smap_source)
            swd       = self._compute_swd(theta_obs, das["theta_fc"], das["theta_wp"])
            rc, label = self._classify_risk(swd)
            scores.append(WatershedDroughtScore(
                watershed_id   = das["id"],
                watershed_name = das["name"],
                theta_obs      = round(theta_obs, 4),
                swd_pct        = round(swd, 2),
                risk_class     = rc,
                risk_label     = label,
                source         = smap_source,
            ))

        warning_count   = sum(1 for s in scores if s.risk_class >= 1)
        emergency_count = sum(1 for s in scores if s.risk_class == 3)
        imbalanced      = [s.watershed_id for s in scores if s.risk_class >= 2]

        status = "degraded" if smap_source == "climatological" else "ok"

        # --- emit Prometheus metrics ---
        self._emit_metrics(scores, warning_count)

        # --- write JSON output ---
        month_str = assessment_date.strftime("%Y%m")
        out_path  = self.OUTPUT_DIR / f"drought_risk_{month_str}.json"
        result    = DroughtRiskResult(
            assessment_date       = assessment_date.isoformat(),
            run_ts                = run_ts,
            watersheds            = scores,
            warning_count         = warning_count,
            emergency_count       = emergency_count,
            imbalanced_watersheds = imbalanced,
            output_path           = str(out_path),
            status                = status,
            notes                 = notes,
        )

        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(
            "Drought risk assessment | date=%s status=%s warning=%d emergency=%d path=%s",
            assessment_date, status, warning_count, emergency_count, out_path,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_smap(self) -> tuple[dict[str, Any], list[str], str]:
        """
        Returns (smap_data, warnings, source).
        source is "smap" or "climatological".
        """
        notes: list[str] = []
        if not self.SMAP_PATH.exists():
            notes.append(
                f"SMAP file not found at {self.SMAP_PATH}; "
                "falling back to climatological mean θ per DAS."
            )
            return {}, notes, "climatological"
        try:
            with open(self.SMAP_PATH) as f:
                data = json.load(f)
            if not data:
                notes.append("SMAP file is empty; falling back to climatological mean.")
                return {}, notes, "climatological"
            return data, notes, "smap"
        except json.JSONDecodeError as exc:
            notes.append(f"SMAP file parse error ({exc}); falling back to climatological mean.")
            return {}, notes, "climatological"

    @staticmethod
    def _get_theta(
        das:        dict[str, Any],
        smap_data:  dict[str, Any],
        source:     str,
    ) -> float:
        """
        Extract root-zone θ for a DAS from SMAP data or climatological mean.
        SMAP JSON expected shape: { "<das_id>": {"theta_root_zone": <float>}, ... }
        """
        if source == "smap":
            entry = smap_data.get(das["id"], {})
            theta = entry.get("theta_root_zone") if isinstance(entry, dict) else None
            if theta is not None:
                return float(theta)
        return das["theta_clim"]

    @staticmethod
    def _compute_swd(theta_obs: float, theta_fc: float, theta_wp: float) -> float:
        """
        Soil Water Deficit [%] = (θ_fc − θ_obs) / (θ_fc − θ_wp) × 100.
        Clipped to [0, 100].
        """
        denom = theta_fc - theta_wp
        if denom <= 0:
            return 0.0
        swd = (theta_fc - theta_obs) / denom * 100.0
        return max(0.0, min(100.0, swd))

    @staticmethod
    def _classify_risk(swd_pct: float) -> tuple[int, str]:
        for threshold, rc, label in _RISK_THRESHOLDS:
            if swd_pct >= threshold:
                return rc, label
        return 0, "NORMAL"

    def _emit_metrics(
        self,
        scores:        list[WatershedDroughtScore],
        warning_count: int,
    ) -> None:
        try:
            from src.hydrology.metrics import (
                DROUGHT_RISK_CLASS,
                DROUGHT_WATERSHEDS_WARNING,
            )
            for s in scores:
                DROUGHT_RISK_CLASS.labels(watershed_id=s.watershed_id).set(s.risk_class)
            DROUGHT_WATERSHEDS_WARNING.set(warning_count)
        except Exception as exc:  # pragma: no cover
            logger.debug("Prometheus metrics unavailable: %s", exc)
