"""
Streamflow Forecast Engine — Sprint 3 Deliverable 1
HYDROLOGIS | Tropi Climate Analytics

6–24 hour ensemble streamflow forecasts for 3 priority rivers:
  - Ciliwung   (DKI Jakarta)   WARNING >150 m³/s  EMERGENCY >300 m³/s
  - Brantas    (East Java)     WARNING >400 m³/s  EMERGENCY >800 m³/s
  - Solo/Bengawan Solo (Central Java) WARNING >600 m³/s EMERGENCY >1200 m³/s

Model ensemble:
  - LSTM streamflow (reuse lstm_streamflow_model from ANALYTICA Sprint 0+1)
  - HBV-light bucket routing (Snow/Soil/Response reservoir)

Inputs:
  - QPE fusion output (Sprint 2: QPEFusionPipeline)
  - SMAP root-zone soil moisture
  - Upstream gauge readings (BMKG HIMET API)

Outputs:
  - StreamflowForecast Pydantic model per river per horizon
  - JSON → workspace/output/streamflow/{river_id}_{timestamp}.json
  - MLflow experiment: 'hydrologis_streamflow'
  - Prometheus: tropi_pipeline_last_ingestion_success_timestamp_seconds{pipeline=flood_early_warning_30min}

Run cadence: 30-minute Airflow DAG (flood_early_warning_30min).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.hydrology.metrics import record_ingestion_success

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECAST_HORIZONS_HOURS = [6, 12, 24]   # forecast steps

RIVER_CONFIGS: dict[str, dict] = {
    "ciliwung": {
        "display_name":    "Ciliwung",
        "province":        "DKI Jakarta",
        "catchment_km2":   347.0,
        "mean_annual_q":   38.0,       # m³/s
        "threshold_watch":    75.0,    # m³/s — watch (inferred, 0.5× WARNING)
        "threshold_warning": 150.0,
        "threshold_emergency": 300.0,
        "lag_hours":       3.0,        # basin lag to outlet gauge
        "hbv_beta":        2.0,
        "hbv_fc":          200.0,      # field capacity (mm)
        "hbv_k_fast":      0.35,
        "hbv_k_slow":      0.05,
    },
    "brantas": {
        "display_name":    "Brantas",
        "province":        "Jawa Timur",
        "catchment_km2":  11800.0,
        "mean_annual_q":  230.0,
        "threshold_watch":   200.0,
        "threshold_warning": 400.0,
        "threshold_emergency": 800.0,
        "lag_hours":       8.0,
        "hbv_beta":        1.8,
        "hbv_fc":          220.0,
        "hbv_k_fast":      0.30,
        "hbv_k_slow":      0.04,
    },
    "solo": {
        "display_name":    "Bengawan Solo",
        "province":        "Jawa Tengah",
        "catchment_km2":  16100.0,
        "mean_annual_q":  310.0,
        "threshold_watch":   300.0,
        "threshold_warning": 600.0,
        "threshold_emergency": 1200.0,
        "lag_hours":      12.0,
        "hbv_beta":        1.6,
        "hbv_fc":          250.0,
        "hbv_k_fast":      0.28,
        "hbv_k_slow":      0.03,
    },
}

MLFLOW_EXPERIMENT = "hydrologis_streamflow"
OUTPUT_DIR = os.path.join(
    os.getenv("WORKSPACE_ROOT", "workspace"), "output", "streamflow"
)

# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class FloodStage(str, Enum):
    NORMAL    = "NORMAL"
    WATCH     = "WATCH"
    WARNING   = "WARNING"
    EMERGENCY = "EMERGENCY"


class StreamflowForecast(BaseModel):
    river_id:                str
    river_name:              str
    province:                str
    forecast_horizon_hours:  int
    discharge_m3s:           list[float] = Field(
        ..., description="Hourly discharge forecast (m³/s) over horizon"
    )
    peak_discharge_m3s:      float
    flood_stage:             FloodStage
    confidence_interval_90:  Tuple[float, float] = Field(
        ..., description="90% CI (lower, upper) for peak discharge (m³/s)"
    )
    model_ensemble:          str = "LSTM+HBV"
    lstm_weight:             float
    hbv_weight:              float
    issued_at:               datetime
    valid_until:             datetime


class ForecastRunStatus(BaseModel):
    run_time_utc:         datetime
    rivers_processed:     int
    forecasts:            list[StreamflowForecast]
    rivers_in_warning:    list[str]
    rivers_in_emergency:  list[str]
    output_paths:         list[str]
    mlflow_run_id:        Optional[str] = None
    success:              bool
    error:                Optional[str] = None


# ---------------------------------------------------------------------------
# HBV-light routing
# ---------------------------------------------------------------------------

class HBVModel:
    """
    Simplified HBV-light bucket model for streamflow routing.

    Reservoirs:
      Snow   : not used (tropical, T always > 0°C)
      Soil   : SM deficit control → effective precipitation
      Response : fast + slow linear reservoirs → runoff

    Reference: Bergström (1995), HBV-96 documentation.
    """

    def __init__(self, river_id: str) -> None:
        cfg = RIVER_CONFIGS[river_id]
        self.beta    = cfg["hbv_beta"]
        self.fc      = cfg["hbv_fc"]      # mm
        self.k_fast  = cfg["hbv_k_fast"]
        self.k_slow  = cfg["hbv_k_slow"]
        self._sm     = cfg["hbv_fc"] * 0.6   # initial soil moisture (mm)
        self._s_fast = 10.0                   # fast reservoir (mm)
        self._s_slow = 20.0                   # slow reservoir (mm)

    def step(
        self,
        precip_mm: float,
        pet_mm: float = 2.0,
        dt_hours: float = 6.0,
    ) -> float:
        """
        Advance model one timestep.

        Returns:
            runoff_mm: Effective runoff (mm) over dt_hours.
        """
        dt_frac = dt_hours / 24.0   # fraction of day

        # Soil moisture accounting
        sm_frac = max(0.0, self._sm / self.fc)
        recharge = precip_mm * (sm_frac ** self.beta)
        self._sm = min(self.fc, self._sm + precip_mm - recharge - pet_mm * dt_frac)
        self._sm = max(0.0, self._sm)

        # Response reservoirs
        self._s_fast += recharge * 0.7
        self._s_slow += recharge * 0.3
        q_fast = self.k_fast * self._s_fast * dt_frac
        q_slow = self.k_slow * self._s_slow * dt_frac
        self._s_fast = max(0.0, self._s_fast - q_fast)
        self._s_slow = max(0.0, self._s_slow - q_slow)

        return q_fast + q_slow   # mm/dt

    def forecast_q_m3s(
        self,
        precip_6h_mm: float,
        catchment_km2: float,
        horizon_hours: int,
        q_init_m3s: float,
    ) -> list[float]:
        """
        Generate hourly discharge forecast over horizon_hours.

        Returns:
            list of discharge values (m³/s), one per hour.
        """
        # Run at 6-hour timestep, interpolate to hourly
        n_steps = max(1, horizon_hours // 6)
        q_steps: list[float] = [q_init_m3s]

        for _ in range(n_steps):
            runoff_mm = self.step(precip_mm=precip_6h_mm, dt_hours=6.0)
            # Convert mm to m³/s: Q = (runoff_mm/1000) * area_m² / (dt_sec)
            q_m3s = (runoff_mm / 1000.0) * (catchment_km2 * 1e6) / (6 * 3600)
            q_steps.append(q_steps[-1] * 0.7 + q_m3s * 0.3 + q_init_m3s * 0.1)

        # Linear interpolation to hourly
        import numpy as np
        x_steps = np.linspace(0, horizon_hours, len(q_steps))
        x_hours = np.arange(1, horizon_hours + 1)
        q_hourly = np.interp(x_hours, x_steps, q_steps)
        return [float(round(q, 2)) for q in q_hourly]


# ---------------------------------------------------------------------------
# LSTM interface (wraps ANALYTICA Sprint 0+1 model)
# ---------------------------------------------------------------------------

class LSTMStreamflowInterface:
    """
    Thin wrapper around ANALYTICA's lstm_streamflow_model.
    Falls back to autoregressive AR(2) stub when model unavailable.
    """

    def __init__(self, river_id: str) -> None:
        self.river_id = river_id
        self._model = None
        try:
            from src.models.lstm_streamflow_model import LSTMStreamflowModel
            self._model = LSTMStreamflowModel(river_id)
            logger.info("LSTMStreamflowModel loaded for %s", river_id)
        except ImportError:
            logger.warning("lstm_streamflow_model not found — using AR stub for %s", river_id)

    def forecast(
        self,
        q_init_m3s: float,
        precip_mm_6h: float,
        sm_m3m3: float,
        horizon_hours: int,
    ) -> list[float]:
        """Returns hourly discharge forecast list."""
        if self._model is not None:
            return self._model.predict(
                current_q=q_init_m3s,
                precip_mm_6h=precip_mm_6h,
                sm_m3m3=sm_m3m3,
                horizon_hours=horizon_hours,
            )

        # AR(2) stub with precipitation forcing
        cfg = RIVER_CONFIGS[self.river_id]
        rng = np.random.default_rng(int(q_init_m3s * 100) % (2**32))
        phi1, phi2 = 0.70, 0.15
        forcing = precip_mm_6h * cfg["catchment_km2"] * 0.05 / 3600
        q = [q_init_m3s, q_init_m3s]
        for _ in range(horizon_hours):
            noise = float(rng.normal(0, q_init_m3s * 0.03))
            q_next = phi1 * q[-1] + phi2 * q[-2] + forcing + noise
            q.append(max(0.0, q_next))
        return [round(v, 2) for v in q[2:]]


# ---------------------------------------------------------------------------
# Ensemble combiner
# ---------------------------------------------------------------------------

def _combine_ensemble(
    lstm_q: list[float],
    hbv_q: list[float],
    lstm_weight: float = 0.65,
) -> list[float]:
    """Weighted average of LSTM and HBV forecasts, element-wise."""
    hbv_weight = 1.0 - lstm_weight
    return [
        round(lstm_weight * l + hbv_weight * h, 2)
        for l, h in zip(lstm_q, hbv_q)
    ]


def _classify_flood_stage(peak_q: float, river_id: str) -> FloodStage:
    cfg = RIVER_CONFIGS[river_id]
    if peak_q >= cfg["threshold_emergency"]: return FloodStage.EMERGENCY
    if peak_q >= cfg["threshold_warning"]:   return FloodStage.WARNING
    if peak_q >= cfg["threshold_watch"]:     return FloodStage.WATCH
    return FloodStage.NORMAL


def _compute_ci90(peak_q: float, uncertainty_frac: float = 0.18) -> Tuple[float, float]:
    """Approximate 90% CI as ±1.645σ with σ = uncertainty_frac × peak_q."""
    sigma = peak_q * uncertainty_frac
    z90 = 1.645
    lo = max(0.0, round(peak_q - z90 * sigma, 1))
    hi = round(peak_q + z90 * sigma, 1)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class StreamflowForecastEngine:
    """
    Ensemble streamflow forecast engine (LSTM + HBV-light).

    Usage (from Airflow flood_early_warning_30min DAG):
        engine = StreamflowForecastEngine()
        status = engine.run(precip_mm_6h=45.0, sm_m3m3=0.32, current_q={...})
    """

    def __init__(self, rivers: Optional[list[str]] = None) -> None:
        self.rivers = rivers or list(RIVER_CONFIGS.keys())
        self._lstms = {r: LSTMStreamflowInterface(r) for r in self.rivers}
        self._hbvs  = {r: HBVModel(r) for r in self.rivers}
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run(
        self,
        precip_mm_6h: float,
        sm_m3m3: float = 0.30,
        current_q: Optional[dict[str, float]] = None,
        lstm_weight: float = 0.65,
    ) -> ForecastRunStatus:
        run_time = datetime.now(timezone.utc)
        q_obs = current_q or {r: RIVER_CONFIGS[r]["mean_annual_q"] for r in self.rivers}

        all_forecasts: list[StreamflowForecast] = []
        output_paths:  list[str] = []
        warning_rivers:   list[str] = []
        emergency_rivers: list[str] = []
        mlflow_run_id: Optional[str] = None

        try:
            mlflow_run_id = self._init_mlflow(run_time)
        except Exception as exc:
            logger.warning("MLflow init failed (non-fatal): %s", exc)

        for river_id in self.rivers:
            cfg     = RIVER_CONFIGS[river_id]
            q_init  = q_obs.get(river_id, cfg["mean_annual_q"])

            for horizon in FORECAST_HORIZONS_HOURS:
                lstm_q = self._lstms[river_id].forecast(
                    q_init_m3s=q_init,
                    precip_mm_6h=precip_mm_6h,
                    sm_m3m3=sm_m3m3,
                    horizon_hours=horizon,
                )
                hbv_q = self._hbvs[river_id].forecast_q_m3s(
                    precip_6h_mm=precip_mm_6h,
                    catchment_km2=cfg["catchment_km2"],
                    horizon_hours=horizon,
                    q_init_m3s=q_init,
                )
                ensemble_q = _combine_ensemble(lstm_q, hbv_q, lstm_weight)
                peak_q     = max(ensemble_q)
                stage      = _classify_flood_stage(peak_q, river_id)
                ci90       = _compute_ci90(peak_q)

                from datetime import timedelta
                forecast = StreamflowForecast(
                    river_id=river_id,
                    river_name=cfg["display_name"],
                    province=cfg["province"],
                    forecast_horizon_hours=horizon,
                    discharge_m3s=ensemble_q,
                    peak_discharge_m3s=round(peak_q, 2),
                    flood_stage=stage,
                    confidence_interval_90=ci90,
                    lstm_weight=lstm_weight,
                    hbv_weight=round(1.0 - lstm_weight, 2),
                    issued_at=run_time,
                    valid_until=run_time + timedelta(hours=horizon),
                )
                all_forecasts.append(forecast)

                if stage == FloodStage.EMERGENCY and river_id not in emergency_rivers:
                    emergency_rivers.append(river_id)
                elif stage == FloodStage.WARNING and river_id not in warning_rivers:
                    warning_rivers.append(river_id)

            # Persist per-river JSON (latest horizon = 24h)
            path = self._write_output(river_id, all_forecasts, run_time)
            output_paths.append(path)

        # Log to MLflow
        if mlflow_run_id:
            self._log_mlflow(mlflow_run_id, all_forecasts)

        # Metrics
        record_ingestion_success("flood_early_warning_30min")

        return ForecastRunStatus(
            run_time_utc=run_time,
            rivers_processed=len(self.rivers),
            forecasts=all_forecasts,
            rivers_in_warning=warning_rivers,
            rivers_in_emergency=emergency_rivers,
            output_paths=output_paths,
            mlflow_run_id=mlflow_run_id,
            success=True,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_output(
        self,
        river_id: str,
        forecasts: list[StreamflowForecast],
        run_time: datetime,
    ) -> str:
        ts = run_time.strftime("%Y%m%dT%H%M%SZ")
        fname = f"{river_id}_{ts}.json"
        path  = os.path.join(OUTPUT_DIR, fname)
        river_forecasts = [f for f in forecasts if f.river_id == river_id]
        payload = {
            "river_id":   river_id,
            "issued_at":  run_time.isoformat(),
            "forecasts":  [f.model_dump() for f in river_forecasts],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Streamflow forecast written: %s", path)
        return path

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    def _init_mlflow(self, run_time: datetime) -> str:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run = mlflow.start_run(
            run_name=f"streamflow_forecast_{run_time.strftime('%Y%m%dT%H%M%SZ')}",
            tags={"pipeline": "flood_early_warning_30min", "agent": "HYDROLOGIS"},
        )
        return run.info.run_id

    def _log_mlflow(self, run_id: str, forecasts: list[StreamflowForecast]) -> None:
        try:
            import mlflow
            with mlflow.start_run(run_id=run_id):
                for f in forecasts:
                    mlflow.log_metric(
                        f"{f.river_id}_peak_q_{f.forecast_horizon_hours}h",
                        f.peak_discharge_m3s,
                    )
                    mlflow.log_param(f"{f.river_id}_stage_{f.forecast_horizon_hours}h", f.flood_stage)
        except Exception as exc:
            logger.warning("MLflow logging failed (non-fatal): %s", exc)
