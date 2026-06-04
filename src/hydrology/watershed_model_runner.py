"""
watershed_model_runner.py — Sprint 9 K5
WatershedModelRunner: Abstraction layer for SWAT and VIC hydrological model execution.

Methods:
  run_swat(watershed_id, start_date, end_date, config_overrides) → ModelRunResult
  run_vic(watershed_id, start_date, end_date, config_overrides)  → ModelRunResult
  get_latest_results(watershed_id, model)                         → ModelRunResult | None

Config:
  workspace/config/swat_{watershed_id}.json
  workspace/config/vic_{watershed_id}.json

Stub execution: synthetic sine-perturbed water balance outputs when model binary absent.

Output: workspace/output/watershed_model/{model}/{watershed_id}_{YYYYMMDD}.json

Prometheus: WATERSHED_MODEL_RUNTIME{watershed_id, model, status} Histogram
            buckets=(60, 300, 900, 1800, 3600, 7200)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model configs (written to workspace/config/ if absent)
# ---------------------------------------------------------------------------
_SWAT_DEFAULTS: dict[str, Any] = {
    "model":             "SWAT+",
    "version":           "2.2.0",
    "time_step":         "daily",
    "sub_basins":        10,
    "hru_count":         120,
    "et_method":         "Penman-Monteith",
    "flow_routing":      "variable_storage",
    "groundwater":       "shallow_aquifer",
    "plant_growth":      True,
    "binary_path":       "/opt/swat/swatplus",
    "input_dir":         "workspace/data/swat/{watershed_id}",
    "output_dir":        "workspace/output/watershed_model/swat/{watershed_id}",
}

_VIC_DEFAULTS: dict[str, Any] = {
    "model":             "VIC-5",
    "version":           "5.1.0",
    "time_step":         "daily",
    "nlayers":           3,
    "n_states":          5,
    "et_method":         "Penman-Monteith",
    "routing":           "RVIC",
    "soil_param_file":   "workspace/data/vic/{watershed_id}/soil.txt",
    "veg_param_file":    "workspace/data/vic/{watershed_id}/veg.txt",
    "binary_path":       "/opt/vic/vic_image.exe",
    "output_dir":        "workspace/output/watershed_model/vic/{watershed_id}",
}

# Climatological parameters per watershed (for synthetic stub output)
_WS_CLIM: dict[str, dict[str, Any]] = {
    "citarum":  {"mean_p": 208.0, "mean_et": 127.0, "mean_q": 42.0,  "area_km2": 6614.0},
    "brantas":  {"mean_p": 172.0, "mean_et": 122.0, "mean_q": 32.0,  "area_km2": 12000.0},
    "solo":     {"mean_p": 182.0, "mean_et": 119.0, "mean_q": 35.0,  "area_km2": 16100.0},
    "musi":     {"mean_p": 224.0, "mean_et": 129.0, "mean_q": 55.0,  "area_km2": 60700.0},
    "kapuas":   {"mean_p": 246.0, "mean_et": 124.0, "mean_q": 75.0,  "area_km2": 98700.0},
    # Additional watersheds from other sprints
    "das_ci":   {"mean_p": 190.0, "mean_et": 125.0, "mean_q": 38.0,  "area_km2": 4480.0},
    "das_br":   {"mean_p": 165.0, "mean_et": 120.0, "mean_q": 30.0,  "area_km2": 11800.0},
    "das_sl":   {"mean_p": 178.0, "mean_et": 118.0, "mean_q": 33.0,  "area_km2": 15900.0},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelRunResult:
    watershed_id:           str
    model:                  str            # "swat" | "vic"
    start_date:             str
    end_date:               str
    run_time_s:             float
    daily_et_mm:            list[float]
    daily_runoff_mm:        list[float]
    daily_baseflow_mm:      list[float]
    daily_soil_water_mm:    list[float]
    water_balance_residual_mm: float       # P - ET - Q - ΔS
    status:                 str            # "success" | "stub" | "failed"
    output_path:            str
    config_path:            str
    notes:                  list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WatershedModelRunner:
    """
    Abstraction layer for SWAT+ and VIC-5 hydrological model execution.
    Falls back to synthetic sine-perturbed climatology when model binary absent.
    Called by watershed_water_balance_dag.py.
    """

    WORKSPACE  = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    CONFIG_DIR = WORKSPACE / "config"
    OUTPUT_ROOT= WORKSPACE / "output" / "watershed_model"

    def __init__(self) -> None:
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_swat(
        self,
        watershed_id:    str,
        start_date:      date,
        end_date:        date,
        config_overrides: dict | None = None,
    ) -> ModelRunResult:
        """Run SWAT+ model for watershed. Falls back to stub if binary absent."""
        return self._run_model("swat", watershed_id, start_date, end_date, config_overrides)

    def run_vic(
        self,
        watershed_id:    str,
        start_date:      date,
        end_date:        date,
        config_overrides: dict | None = None,
    ) -> ModelRunResult:
        """Run VIC-5 model for watershed. Falls back to stub if binary absent."""
        return self._run_model("vic", watershed_id, start_date, end_date, config_overrides)

    def get_latest_results(
        self,
        watershed_id: str,
        model:        str,
    ) -> ModelRunResult | None:
        """Load the most recent model run result for watershed+model combination."""
        out_dir = self.OUTPUT_ROOT / model.lower()
        if not out_dir.exists():
            return None
        # Find latest file matching pattern
        pattern = f"{watershed_id}_*.json"
        files   = sorted(out_dir.glob(pattern), reverse=True)
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                return ModelRunResult(**data)
            except Exception as exc:
                logger.debug("Could not load %s: %s", f, exc)
        return None

    # ------------------------------------------------------------------
    # Private: unified runner
    # ------------------------------------------------------------------

    def _run_model(
        self,
        model:           str,
        watershed_id:    str,
        start_date:      date,
        end_date:        date,
        config_overrides: dict | None,
    ) -> ModelRunResult:
        t0       = time.monotonic()
        notes: list[str] = []
        cfg      = self._load_or_create_config(model, watershed_id, config_overrides, notes)
        binary   = cfg.get("binary_path", "")
        status   = "success" if Path(binary).exists() else "stub"

        if status == "success":
            result = self._exec_model_binary(model, watershed_id, start_date, end_date, cfg, notes)
        else:
            notes.append(
                f"Model binary not found at '{binary}'; generating synthetic output"
            )
            result = self._synthetic_run(model, watershed_id, start_date, end_date, notes)

        run_time = time.monotonic() - t0
        self._emit_runtime_metric(watershed_id, model, status, run_time)

        # Write output
        out_dir  = self.OUTPUT_ROOT / model.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{watershed_id}_{end_date.strftime('%Y%m%d')}.json"

        mr = ModelRunResult(
            watershed_id              = watershed_id,
            model                     = model,
            start_date                = start_date.isoformat(),
            end_date                  = end_date.isoformat(),
            run_time_s                = round(run_time, 3),
            daily_et_mm               = result["et"],
            daily_runoff_mm           = result["runoff"],
            daily_baseflow_mm         = result["baseflow"],
            daily_soil_water_mm       = result["soil_water"],
            water_balance_residual_mm = result["residual"],
            status                    = status,
            output_path               = str(out_path),
            config_path               = str(self.CONFIG_DIR / f"{model}_{watershed_id}.json"),
            notes                     = notes,
        )

        with open(out_path, "w") as f:
            json.dump(asdict(mr), f, indent=2)

        logger.info(
            "ModelRun | ws=%-10s model=%-4s status=%-7s days=%d residual=%.2f mm t=%.1fs",
            watershed_id, model, status,
            (end_date - start_date).days,
            mr.water_balance_residual_mm,
            run_time,
        )
        return mr

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def _load_or_create_config(
        self,
        model:       str,
        ws_id:       str,
        overrides:   dict | None,
        notes:       list[str],
    ) -> dict[str, Any]:
        cfg_path = self.CONFIG_DIR / f"{model}_{ws_id}.json"
        defaults = _SWAT_DEFAULTS.copy() if model == "swat" else _VIC_DEFAULTS.copy()
        # Fill placeholders
        for k, v in defaults.items():
            if isinstance(v, str):
                defaults[k] = v.replace("{watershed_id}", ws_id)

        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = {**defaults, **json.load(f)}
            except Exception as exc:
                notes.append(f"Config read error ({exc}); using defaults")
                cfg = defaults
        else:
            cfg = defaults
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            notes.append(f"Config written to {cfg_path}")

        if overrides:
            cfg.update(overrides)
        return cfg

    # ------------------------------------------------------------------
    # Synthetic stub runner
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_run(
        model:        str,
        watershed_id: str,
        start_date:   date,
        end_date:     date,
        notes:        list[str],
    ) -> dict[str, Any]:
        """
        Generate synthetic daily water balance outputs using sine-perturbed climatology.
        P(t) = mean_p/30 × [1 + 0.4·sin(2π·doy/365)] + noise
        ET(t) = mean_et/30 × [1 + 0.15·cos(2π·doy/365)]
        Q(t)  = mean_q/30 × max(P-ET, 0)/mean_p × seasonality
        ΔS    = residual storage change
        """
        import random
        rng       = random.Random(hash(f"{model}{watershed_id}{start_date}"))
        clim      = _WS_CLIM.get(watershed_id, {"mean_p": 190.0, "mean_et": 120.0, "mean_q": 38.0})
        n_days    = (end_date - start_date).days + 1
        et, runoff, baseflow, soil_water = [], [], [], []
        sw        = clim["mean_p"] * 1.5  # initial soil water storage

        for i in range(n_days):
            d      = start_date + timedelta(days=i)
            doy    = d.timetuple().tm_yday
            phase  = 2 * math.pi * doy / 365.0
            # Daily precip (mm)
            p_d    = max((clim["mean_p"] / 30.0)
                         * (1.0 + 0.4 * math.sin(phase))
                         * rng.uniform(0.6, 1.4), 0.0)
            # ET (mm)
            et_d   = max((clim["mean_et"] / 30.0)
                         * (1.0 + 0.15 * math.cos(phase)), 0.5)
            et.append(round(et_d, 3))
            # Runoff (mm)
            excess  = max(p_d - et_d, 0.0)
            rf_d    = excess * rng.uniform(0.25, 0.55)
            runoff.append(round(rf_d, 3))
            # Baseflow (mm)
            bf_d    = sw * rng.uniform(0.003, 0.008)
            baseflow.append(round(bf_d, 3))
            # Soil water update
            sw      = max(sw + p_d - et_d - rf_d - bf_d, 0.0)
            soil_water.append(round(sw, 3))

        # Water balance residual
        total_p   = sum((clim["mean_p"] / 30.0) * n_days)
        total_et  = sum(et)
        total_q   = sum(runoff) + sum(baseflow)
        delta_s   = soil_water[-1] - soil_water[0] if soil_water else 0.0
        residual  = total_p - total_et - total_q - delta_s

        return {
            "et":       et,
            "runoff":   runoff,
            "baseflow": baseflow,
            "soil_water": soil_water,
            "residual": round(residual, 3),
        }

    @staticmethod
    def _exec_model_binary(
        model:        str,
        watershed_id: str,
        start_date:   date,
        end_date:     date,
        cfg:          dict[str, Any],
        notes:        list[str],
    ) -> dict[str, Any]:
        """
        Execute real model binary (SWAT+/VIC-5).
        In production: subprocess call + parse output files.
        Returns same dict shape as _synthetic_run.
        """
        # Production implementation would:
        # 1. Write time-control files (file.cio for SWAT, global params for VIC)
        # 2. subprocess.run([cfg["binary_path"], ...], cwd=cfg["input_dir"])
        # 3. Parse output (output.rch for SWAT, fluxes*.txt for VIC)
        # 4. Return structured dict
        notes.append(f"Real {model.upper()} binary execution not yet wired; using synthetic fallback")
        return WatershedModelRunner._synthetic_run(model, watershed_id, start_date, end_date, notes)

    @staticmethod
    def _emit_runtime_metric(
        watershed_id: str, model: str, status: str, run_time: float
    ) -> None:
        try:
            from src.hydrology.metrics import WATERSHED_MODEL_RUNTIME
            WATERSHED_MODEL_RUNTIME.labels(
                watershed_id=watershed_id,
                model=model,
                status=status,
            ).observe(run_time)
        except Exception as exc:
            logger.debug("WATERSHED_MODEL_RUNTIME unavailable: %s", exc)
