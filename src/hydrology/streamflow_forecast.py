"""
streamflow_forecast.py — Sprint 6 E1
StreamflowForecaster: Rational-method Q = C·i·A streamflow forecasting for
Ciliwung, Brantas, and Solo rivers across sub-watersheds.

Inputs:
  workspace/data/qpe_latest_validated.json  — QPE precipitation grid
  workspace/data/bmkg_gauge_latest.json     — BMKG gauge observations

Outputs:
  workspace/output/streamflow/forecast_{river_id}_{YYYYMMDD_HHMM}.json

Prometheus:
  tropi_streamflow_forecast_peak_cms{river_id, horizon_hr}  Gauge
  tropi_streamflow_forecast_runs_total{river_id, status}    Counter
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
# River / sub-watershed catalogue
# ---------------------------------------------------------------------------

# Rational method parameters per sub-watershed:
#   C  — runoff coefficient (dimensionless, 0–1)
#   A  — drainage area (km²)
# Reference: PUPR DAS Strategis Nasional 2023 inventory
_RIVER_CATALOGUE: dict[str, list[dict[str, Any]]] = {
    "ciliwung": [
        {"sub_id": "CI-UP",  "C": 0.65, "A_km2": 149.0},
        {"sub_id": "CI-MID", "C": 0.72, "A_km2": 211.0},
        {"sub_id": "CI-LOW", "C": 0.80, "A_km2": 87.0},
    ],
    "brantas": [
        {"sub_id": "BR-UP",  "C": 0.55, "A_km2": 2480.0},
        {"sub_id": "BR-MID", "C": 0.60, "A_km2": 5310.0},
        {"sub_id": "BR-LOW", "C": 0.68, "A_km2": 2400.0},
    ],
    "solo": [
        {"sub_id": "SL-UP",  "C": 0.52, "A_km2": 3780.0},
        {"sub_id": "SL-MID", "C": 0.58, "A_km2": 6520.0},
        {"sub_id": "SL-LOW", "C": 0.65, "A_km2": 4200.0},
    ],
}

# Time-of-concentration (hours) per river for lag routing
_TC_HOURS: dict[str, float] = {
    "ciliwung": 3.5,
    "brantas":  8.0,
    "solo":     12.0,
}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HorizonForecast:
    horizon_hr:   int
    peak_cms:     float          # peak discharge (m³/s)
    volume_Mm3:   float          # total runoff volume (million m³)
    lag_hrs:      float          # routing lag to outlet
    confidence:   float          # 0–1, based on QPE coverage quality


@dataclass
class StreamflowForecastResult:
    river_id:       str
    run_ts:         str          # ISO-8601 UTC
    horizons:       list[HorizonForecast]
    peak_cms_max:   float        # max across all horizons
    data_sources:   dict[str, Any]
    output_path:    str
    status:         str          # "ok" | "degraded" | "skipped"
    warnings:       list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StreamflowForecaster:
    """
    Rational-method streamflow forecaster for Ciliwung, Brantas, and Solo.

    Q = C · i · A / 3.6
      where i is mean rainfall intensity (mm/hr) over the contributing area
      and A is drainage area (km²) → Q in m³/s.

    Multi-horizon: each horizon aggregates QPE rainfall over that window,
    routing it through a simple lag model (tc = time-of-concentration).
    """

    WORKSPACE   = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    DATA_DIR    = WORKSPACE / "data"
    OUTPUT_DIR  = WORKSPACE / "output" / "streamflow"
    QPE_PATH    = DATA_DIR / "qpe_latest_validated.json"
    GAUGE_PATH  = DATA_DIR / "bmkg_gauge_latest.json"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        river_id: str,
        horizons: list[int] | None = None,
    ) -> StreamflowForecastResult:
        """
        Forecast streamflow for *river_id* at each horizon in *horizons* hours.
        Returns StreamflowForecastResult with per-horizon peak discharge values.
        """
        if horizons is None:
            horizons = [6, 12, 24]

        river_id = river_id.lower()
        run_ts   = datetime.now(timezone.utc).isoformat()

        if river_id not in _RIVER_CATALOGUE:
            raise ValueError(
                f"Unknown river_id '{river_id}'. "
                f"Supported: {list(_RIVER_CATALOGUE.keys())}"
            )

        warnings: list[str] = []
        status = "ok"

        # --- load QPE ---
        qpe_data, qpe_warn = self._load_qpe()
        warnings.extend(qpe_warn)

        # --- load gauge (for QPE bias correction) ---
        gauge_data, gauge_warn = self._load_gauge(river_id)
        warnings.extend(gauge_warn)

        # --- QPE bias correction factor from gauge ---
        bias_factor = self._compute_bias_factor(river_id, qpe_data, gauge_data)

        # --- sub-watershed forecasts ---
        sub_watersheds = _RIVER_CATALOGUE[river_id]
        tc             = _TC_HOURS[river_id]

        horizon_results: list[HorizonForecast] = []
        for h in sorted(horizons):
            peak_cms, volume_Mm3, conf = self._forecast_horizon(
                sub_watersheds, qpe_data, h, tc, bias_factor
            )
            horizon_results.append(
                HorizonForecast(
                    horizon_hr  = h,
                    peak_cms    = round(peak_cms, 2),
                    volume_Mm3  = round(volume_Mm3, 4),
                    lag_hrs     = round(tc * 0.6, 2),   # simplified lag = 0.6 · tc
                    confidence  = round(conf, 3),
                )
            )

        peak_max = max(h.peak_cms for h in horizon_results)

        if warnings:
            status = "degraded"

        # --- emit Prometheus metrics ---
        self._emit_metrics(river_id, horizon_results, status)

        # --- write output ---
        ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        out_path  = self.OUTPUT_DIR / f"forecast_{river_id}_{ts_str}.json"
        result    = StreamflowForecastResult(
            river_id     = river_id,
            run_ts       = run_ts,
            horizons     = horizon_results,
            peak_cms_max = round(peak_max, 2),
            data_sources = {
                "qpe_path":   str(self.QPE_PATH),
                "gauge_path": str(self.GAUGE_PATH),
            },
            output_path  = str(out_path),
            status       = status,
            warnings     = warnings,
        )

        with open(out_path, "w") as f:
            json.dump(self._result_to_dict(result), f, indent=2)

        logger.info(
            "Streamflow forecast | river=%s status=%s peak_max=%.1f m³/s path=%s",
            river_id, status, peak_max, out_path,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_qpe(self) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        if not self.QPE_PATH.exists():
            warnings.append(f"QPE file not found at {self.QPE_PATH}; using zero precipitation")
            return {"mean_precip_mm_hr": 0.0, "coverage_pct": 0.0, "valid": False}, warnings
        with open(self.QPE_PATH) as f:
            data = json.load(f)
        if not data.get("valid", True):
            warnings.append("QPE file marked invalid; results degraded")
        return data, warnings

    def _load_gauge(
        self, river_id: str
    ) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        if not self.GAUGE_PATH.exists():
            warnings.append(f"Gauge file not found at {self.GAUGE_PATH}; skipping bias correction")
            return {}, warnings
        with open(self.GAUGE_PATH) as f:
            data = json.load(f)
        river_gauges = data.get(river_id, {})
        if not river_gauges:
            warnings.append(f"No gauge data for river '{river_id}' in {self.GAUGE_PATH}")
        return river_gauges, warnings

    def _compute_bias_factor(
        self,
        river_id: str,
        qpe_data:   dict[str, Any],
        gauge_data: dict[str, Any],
    ) -> float:
        """
        Ratio of mean gauge precip to QPE precip for the river.
        Returns 1.0 (no correction) when data is unavailable.
        """
        qpe_precip   = float(qpe_data.get("mean_precip_mm_hr", 0.0) or 0.0)
        # gauge data may store mean_precip_mm_hr at river level
        gauge_precip = float(gauge_data.get("mean_precip_mm_hr", 0.0) or 0.0)
        if qpe_precip <= 0 or gauge_precip <= 0:
            return 1.0
        factor = gauge_precip / qpe_precip
        # Clip to a reasonable range [0.5, 2.0] to avoid outlier inflation
        return max(0.5, min(2.0, factor))

    def _forecast_horizon(
        self,
        sub_watersheds: list[dict[str, Any]],
        qpe_data:       dict[str, Any],
        horizon_hr:     int,
        tc:             float,
        bias_factor:    float,
    ) -> tuple[float, float, float]:
        """
        Compute basin-aggregated peak Q (m³/s), runoff volume (Mm³),
        and confidence score for a given forecast horizon.
        """
        # Mean QPE intensity over the horizon window (mm/hr)
        # For multi-horizon: intensity decays with a simple exp envelope
        base_intensity = float(qpe_data.get("mean_precip_mm_hr", 0.0) or 0.0)
        base_intensity *= bias_factor

        # Exponential decay factor: longer horizons see lower sustained intensity
        decay = math.exp(-0.03 * max(0, horizon_hr - 6))
        i_eff = base_intensity * decay   # effective intensity (mm/hr)

        # Rational method aggregated across sub-watersheds:
        # Q_peak = sum(C_j · i · A_j) / 3.6    [m³/s]
        q_peak    = 0.0
        total_A   = 0.0
        for sw in sub_watersheds:
            q_peak  += sw["C"] * i_eff * sw["A_km2"] / 3.6
            total_A += sw["A_km2"]

        # Routing: time-shift peak by lag (0.6 · tc); if horizon < lag, Q is pre-peak
        lag = 0.6 * tc
        if horizon_hr < lag:
            ratio  = horizon_hr / lag
            q_peak = q_peak * ratio

        # Runoff volume = Q_peak · duration (simplified rectangular hydrograph)
        # V [m³] = Q_peak [m³/s] · horizon [s]; convert to Mm³
        v_Mm3 = q_peak * horizon_hr * 3600 / 1e6

        # Confidence: penalise low QPE coverage and long horizons
        coverage = float(qpe_data.get("coverage_pct", 100.0) or 100.0)
        conf = (coverage / 100.0) * math.exp(-0.015 * horizon_hr)

        return q_peak, v_Mm3, conf

    def _emit_metrics(
        self,
        river_id: str,
        horizons: list[HorizonForecast],
        status:   str,
    ) -> None:
        try:
            from src.hydrology.metrics import (
                STREAMFLOW_FORECAST_PEAK,
                STREAMFLOW_FORECAST_RUNS,
            )
            for h in horizons:
                STREAMFLOW_FORECAST_PEAK.labels(
                    river_id=river_id,
                    horizon_hr=str(h.horizon_hr),
                ).set(h.peak_cms)
            STREAMFLOW_FORECAST_RUNS.labels(
                river_id=river_id,
                status=status,
            ).inc()
        except Exception as exc:  # pragma: no cover
            logger.debug("Prometheus metrics unavailable: %s", exc)

    @staticmethod
    def _result_to_dict(result: StreamflowForecastResult) -> dict[str, Any]:
        d = asdict(result)
        return d
