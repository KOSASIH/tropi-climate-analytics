"""
streamflow_forecast.py — Sprint 8 J5 (replaces Sprint 6 version)
StreamflowForecaster: 6-24hr streamflow forecast for Ciliwung, Brantas, Solo
using LSTM model outputs + QPE latest input; ensemble of 3 lead times.

Model input:
  latest_qpe.json       → antecedent precipitation
  workspace/data/smap/  → soil moisture state
  LSTM stub: power-law persistence model h(t+Δ) = h(t) × (Q_forecast/Q_base)^0.45

Outputs:
  workspace/output/streamflow/forecast_{river_id}_{YYYYMMDD_HHMM}.json
  workspace/output/streamflow/latest_forecast.json  (rolling sidecar → ANALYTICA)

Flood threshold exceedance: checks against FLOOD_THRESHOLDS (metres, bankfull)

Prometheus (Sprint 8): STREAMFLOW_FORECAST_BIAS{river_id, horizon_hr} Gauge
                       (updated after verification; set to 0.0 at issue time)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# River catalogue
# ---------------------------------------------------------------------------
_RIVERS: dict[str, dict[str, Any]] = {
    "ciliwung": {
        "name":           "Ciliwung",
        "area_km2":       447.0,
        "base_cms":       45.0,       # mean annual baseflow
        "bankfull_cms":   280.0,      # bankfull discharge
        "thresholds_m":   {"WATCH": 1.5, "WARNING": 2.5, "EMERGENCY": 4.0},
        "rating_a":       12.4,
        "rating_b":       0.68,
        "tc_hr":          3.0,        # time of concentration
    },
    "brantas": {
        "name":           "Brantas",
        "area_km2":       12000.0,
        "base_cms":       280.0,
        "bankfull_cms":   1800.0,
        "thresholds_m":   {"WATCH": 3.0, "WARNING": 5.0, "EMERGENCY": 8.0},
        "rating_a":       28.1,
        "rating_b":       0.72,
        "tc_hr":          12.0,
    },
    "solo": {
        "name":           "Solo",
        "area_km2":       16100.0,
        "base_cms":       380.0,
        "bankfull_cms":   2500.0,
        "thresholds_m":   {"WATCH": 4.0, "WARNING": 6.5, "EMERGENCY": 10.0},
        "rating_a":       35.6,
        "rating_b":       0.74,
        "tc_hr":          18.0,
    },
}

# Lead times for ensemble
_HORIZONS_HR = [6, 12, 24]

# Power-law persistence exponent (LSTM stub)
_PERSISTENCE_EXP = 0.45


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HorizonForecast:
    horizon_hr:  int
    forecast_cms: float
    stage_m:     float
    flood_stage: str     # "NORMAL" | "WATCH" | "WARNING" | "EMERGENCY"


@dataclass
class StreamflowForecast:
    river_id:                    str
    river_name:                  str
    issue_time:                  str          # ISO 8601
    horizon_hr:                  int          # maximum horizon
    forecast_cms_6hr:            float
    forecast_cms_12hr:           float
    forecast_cms_24hr:           float
    confidence_pct:              float        # 0–100
    flood_threshold_exceedance:  dict[str, bool]
    horizons:                    list[HorizonForecast]
    antecedent_precip_mm:        float
    soil_moisture_theta:         float
    model_type:                  str          # "lstm" | "persistence"
    output_path:                 str
    sidecar_path:                str
    notes:                       list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StreamflowForecaster:
    """
    Multi-horizon streamflow forecaster for Ciliwung, Brantas, and Solo.
    Uses LSTM model when available; falls back to power-law persistence.
    """

    WORKSPACE      = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    SMAP_DIR       = WORKSPACE / "data" / "smap"
    QPE_SIDECAR    = WORKSPACE / "output" / "qpe" / "latest_qpe.json"
    OUTPUT_DIR     = WORKSPACE / "output" / "streamflow"
    SIDECAR        = WORKSPACE / "output" / "streamflow" / "latest_forecast.json"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        river_id:    str,
        issue_time:  datetime | None = None,
        horizon_hr:  int = 24,
    ) -> StreamflowForecast:
        """
        Issue multi-horizon streamflow forecast for *river_id*.
        Returns StreamflowForecast with 6hr / 12hr / 24hr ensemble members.
        """
        if issue_time is None:
            issue_time = datetime.now(timezone.utc)

        river_id = river_id.lower()
        if river_id not in _RIVERS:
            raise ValueError(
                f"Unknown river_id '{river_id}'. Supported: {list(_RIVERS.keys())}"
            )

        river  = _RIVERS[river_id]
        notes: list[str] = []

        # --- Load QPE antecedent precipitation ---
        ant_precip = self._load_qpe_precip(notes)

        # --- Load SMAP soil moisture ---
        sm_theta = self._load_smap_theta(river_id, notes)

        # --- Current discharge estimate (rating curve from QPE proxy) ---
        Q_current = self._estimate_current_discharge(river, ant_precip, sm_theta)

        # --- Ensemble forecasts at each horizon ---
        horizon_results: list[HorizonForecast] = []
        cms_by_hr: dict[int, float] = {}

        for hr in _HORIZONS_HR:
            if hr > horizon_hr:
                # Pad with last valid horizon
                q = cms_by_hr.get(max(cms_by_hr.keys(), default=hr), Q_current)
            else:
                q = self._persistence_forecast(Q_current, river, ant_precip, sm_theta, hr)
            stage = self._discharge_to_stage(q, river)
            fs    = self._classify_stage(stage, river)
            horizon_results.append(HorizonForecast(
                horizon_hr=hr, forecast_cms=round(q, 2), stage_m=round(stage, 3), flood_stage=fs
            ))
            cms_by_hr[hr] = q

        q_6  = cms_by_hr.get(6,  Q_current)
        q_12 = cms_by_hr.get(12, Q_current)
        q_24 = cms_by_hr.get(24, Q_current)

        # --- Flood threshold exceedance ---
        peak_q  = max(q_6, q_12, q_24)
        peak_st = self._discharge_to_stage(peak_q, river)
        exceedance = {
            lvl: peak_st >= thr
            for lvl, thr in river["thresholds_m"].items()
        }

        # --- Confidence estimate ---
        confidence = self._estimate_confidence(ant_precip, sm_theta, horizon_hr)

        # --- Write output ---
        ts_str   = issue_time.strftime("%Y%m%d_%H%M")
        out_path = self.OUTPUT_DIR / f"forecast_{river_id}_{ts_str}.json"

        result = StreamflowForecast(
            river_id                   = river_id,
            river_name                 = river["name"],
            issue_time                 = issue_time.isoformat(),
            horizon_hr                 = horizon_hr,
            forecast_cms_6hr           = round(q_6,  2),
            forecast_cms_12hr          = round(q_12, 2),
            forecast_cms_24hr          = round(q_24, 2),
            confidence_pct             = round(confidence, 1),
            flood_threshold_exceedance = exceedance,
            horizons                   = horizon_results,
            antecedent_precip_mm       = round(ant_precip, 3),
            soil_moisture_theta        = round(sm_theta, 4),
            model_type                 = "persistence",
            output_path                = str(out_path),
            sidecar_path               = str(self.SIDECAR),
            notes                      = notes,
        )

        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        # Overwrite rolling sidecar (ANALYTICA training feedback)
        with open(self.SIDECAR, "w") as f:
            json.dump(asdict(result), f, indent=2)

        # Prometheus bias initialised at 0.0 at issue time
        self._emit_bias_metric(river_id, horizon_hr, 0.0)

        logger.info(
            "Forecast | river=%-9s t+6h=%.1f t+12h=%.1f t+24h=%.1f m³/s "
            "conf=%.0f%% exceedances=%s",
            river_id, q_6, q_12, q_24, confidence,
            [k for k, v in exceedance.items() if v] or "none",
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_qpe_precip(self, notes: list[str]) -> float:
        """Load max precip from latest QPE sidecar (antecedent input)."""
        if self.QPE_SIDECAR.exists():
            try:
                with open(self.QPE_SIDECAR) as f:
                    data = json.load(f)
                return float(data.get("max_precip_mm", 0.0))
            except Exception as exc:
                notes.append(f"QPE sidecar read error ({exc}); antecedent precip = 0")
        else:
            notes.append("QPE sidecar absent; antecedent precip = 0")
        return 0.0

    def _load_smap_theta(self, river_id: str, notes: list[str]) -> float:
        """Load root-zone soil moisture θ from SMAP latest file."""
        path = self.WORKSPACE / "data" / "smap_latest.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                # Map river to DAS
                river_to_das = {
                    "ciliwung": "DAS-CI", "brantas": "DAS-BR", "solo": "DAS-SL"
                }
                das  = river_to_das.get(river_id, river_id.upper())
                entry= data.get(das, {})
                if isinstance(entry, dict):
                    return float(entry.get("theta_root_zone", 0.30) or 0.30)
            except Exception as exc:
                notes.append(f"SMAP theta read error ({exc}); θ = 0.30")
        return 0.30  # climatological fallback

    @staticmethod
    def _estimate_current_discharge(
        river:      dict[str, Any],
        precip_mm:  float,
        sm_theta:   float,
    ) -> float:
        """
        Estimate current discharge using rational-method proxy:
        Q = C × i × A / 360
        C (runoff coefficient) scales with SM saturation level.
        """
        C     = min(0.3 + (sm_theta / 0.4) * 0.4, 0.9)
        i     = precip_mm / 0.5        # mm/30min → mm/hr
        A_ha  = river["area_km2"] * 100.0
        Q_m3s = C * i * A_ha / 360.0  # rational formula
        return max(Q_m3s, river["base_cms"] * 0.5)

    @staticmethod
    def _persistence_forecast(
        Q_current:  float,
        river:      dict[str, Any],
        precip_mm:  float,
        sm_theta:   float,
        horizon_hr: int,
    ) -> float:
        """
        LSTM stub: power-law persistence model
        h(t+Δ) = h(t) × (Q_forecast / Q_base)^0.45

        Applies exponential decay toward baseflow with QPE-driven peak enhancement.
        """
        base   = river["base_cms"]
        tc     = river["tc_hr"]
        # Precipitation-driven peak response within time of concentration
        if horizon_hr <= tc:
            # Rising limb: peak driven by QPE intensity
            C      = min(0.3 + (sm_theta / 0.4) * 0.45, 0.9)
            i      = precip_mm * 2.0 / horizon_hr  # mm/hr average intensity
            A_ha   = river["area_km2"] * 100.0
            Q_peak = C * i * A_ha / 360.0
            alpha  = horizon_hr / tc
            Q_fct  = Q_current + alpha * max(Q_peak - Q_current, 0.0)
        else:
            # Recession limb: exponential decay toward baseflow
            k    = math.exp(-math.log(2) / tc)    # half-life = tc
            dt   = horizon_hr - tc
            Q_fct= base + (Q_current - base) * (k ** dt)

        # Power-law persistence scaling
        Q_out = Q_fct * (max(Q_fct, 0.01) / max(base, 0.01)) ** _PERSISTENCE_EXP
        return max(round(Q_out, 2), base * 0.1)

    @staticmethod
    def _discharge_to_stage(q_cms: float, river: dict[str, Any]) -> float:
        """Convert discharge to stage using Manning power-law rating curve h = a·Q^b."""
        a = river["rating_a"]
        b = river["rating_b"]
        return a * (max(q_cms, 0.01) ** b)

    @staticmethod
    def _classify_stage(stage_m: float, river: dict[str, Any]) -> str:
        thresholds = river["thresholds_m"]
        if stage_m >= thresholds["EMERGENCY"]:
            return "EMERGENCY"
        if stage_m >= thresholds["WARNING"]:
            return "WARNING"
        if stage_m >= thresholds["WATCH"]:
            return "WATCH"
        return "NORMAL"

    @staticmethod
    def _estimate_confidence(
        ant_precip:  float,
        sm_theta:    float,
        horizon_hr:  int,
    ) -> float:
        """
        Heuristic confidence: decreases with longer horizon and high precip uncertainty.
        base = 85% at 6hr, ~65% at 24hr; reduced by precip intensity.
        """
        base = max(90.0 - horizon_hr * 1.1, 50.0)
        if ant_precip > 20.0:
            base -= 10.0  # high precip → more uncertain
        if sm_theta > 0.35:
            base -= 5.0   # near-saturated → nonlinear response
        return max(min(base, 95.0), 40.0)

    @staticmethod
    def _emit_bias_metric(river_id: str, horizon_hr: int, bias_cms: float) -> None:
        try:
            from src.hydrology.metrics import STREAMFLOW_FORECAST_BIAS
            STREAMFLOW_FORECAST_BIAS.labels(
                river_id=river_id,
                horizon_hr=str(horizon_hr),
            ).set(bias_cms)
        except Exception as exc:
            logger.debug("STREAMFLOW_FORECAST_BIAS unavailable: %s", exc)
