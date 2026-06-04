"""
reservoir_operations.py — Sprint 10 M3
ReservoirOperationsModel: Rule-curve-based reservoir routing for 5 major Indonesian reservoirs.

Reservoirs: jatiluhur, saguling, cirata, gajah_mungkur, upper_cisokan

Rule-curve levels (fraction of active capacity):
  MOL (dead):   0.05 — no release
  TOL (target): 0.60 — normal operations
  FSL:          0.90 — pre-release to recover headroom; DOWNSTREAM alert
  MWL:          1.00 — emergency spill; max gate; BPBD dispatch

Outputs:
  workspace/output/reservoir/reservoir_{reservoir_id}_{YYYYMMDD_HH}.json
  workspace/output/reservoir/active_operations.json (rolling sidecar — GEOSPATIAL + VISUALIA)

Prometheus: RESERVOIR_STORAGE_FRACTION{reservoir_id} Gauge
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule curve fractions of active capacity
# ---------------------------------------------------------------------------
_MOL = 0.05
_TOL = 0.60
_FSL = 0.90
_MWL = 1.00

# Gate status labels
_GATE_NORMAL    = "NORMAL"
_GATE_ALERT     = "ALERT"
_GATE_EMERGENCY = "EMERGENCY"

ALL_RESERVOIRS = ["jatiluhur", "saguling", "cirata", "gajah_mungkur", "upper_cisokan"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StorageUpdate:
    reservoir_id:      str
    reservoir_name:    str
    timestamp:         str
    storage_m3:        float
    active_fraction:   float   # fraction of active_capacity_m3
    level_masl:        float
    inflow_cms:        float
    outflow_cms:       float
    gate_status:       str     # NORMAL | ALERT | EMERGENCY
    water_supply_risk: bool
    output_path:       str
    notes:             list[str] = field(default_factory=list)


@dataclass
class FloodGateDecision:
    reservoir_id:          str
    reservoir_name:        str
    recommended_release_cms: float
    gate_status:           str
    rationale:             str
    dispatch_alert:        bool
    alert_level:           str
    forecast_horizon_hr:   int
    peak_inflow_cms:       float
    output_path:           str
    notes:                 list[str] = field(default_factory=list)


@dataclass
class DownstreamImpact:
    reservoir_id:       str
    release_cms:        float
    peak_cms_24hr:      float   # estimated peak discharge 24hr downstream
    travel_time_hr:     float
    affected_cities:    list[str]
    impact_level:       str     # LOW | MODERATE | HIGH | CRITICAL


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReservoirOperationsModel:
    """
    Rule-curve reservoir routing for 5 major Indonesian reservoirs.
    Reads config from workspace/config/reservoir_config.json.
    """

    WORKSPACE    = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    CONFIG_FILE  = WORKSPACE / "config" / "reservoir_config.json"
    OUTPUT_DIR   = WORKSPACE / "output" / "reservoir"
    ACTIVE_OPS   = WORKSPACE / "output" / "reservoir" / "active_operations.json"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()
        self._storage_state: dict[str, float] = {}  # in-memory storage m3 cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_storage(
        self,
        reservoir_id:  str,
        inflow_cms:    float,
        outflow_cms:   float,
        dt_hours:      float = 1.0,
        timestamp:     datetime | None = None,
    ) -> StorageUpdate:
        """
        Update reservoir storage by routing inflow - outflow over dt_hours.
        Returns StorageUpdate with current state and gate status classification.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        cfg    = self._get_cfg(reservoir_id)
        name   = cfg["name"]
        cap    = cfg["active_capacity_m3"]
        dead   = cfg["dead_storage_m3"]
        fsl_m  = cfg["FSL_masl"]
        mwl_m  = cfg["MWL_masl"]
        notes: list[str] = []

        # Current storage
        curr_s = self._storage_state.get(reservoir_id, cap * 0.65)
        # Route: ΔS = (inflow - outflow) × dt_sec
        delta_s = (inflow_cms - outflow_cms) * dt_hours * 3600.0
        new_s   = max(min(curr_s + delta_s, cap + dead), dead)
        self._storage_state[reservoir_id] = new_s

        frac    = (new_s - dead) / max(cap, 1.0)
        # Level interpolation: linear between dead (FSL-10) and MWL
        level   = fsl_m - 10.0 + (frac / _FSL) * 10.0
        level   = min(level, mwl_m)

        # Gate status
        if frac >= _MWL:
            gate_status = _GATE_EMERGENCY
            notes.append("MWL breach — emergency spill; BPBD dispatch triggered")
        elif frac >= _FSL:
            gate_status = _GATE_ALERT
            notes.append("Above FSL — pre-releasing to recover headroom; downstream alert")
        else:
            gate_status = _GATE_NORMAL

        water_supply_risk = frac <= _MOL

        # Prometheus
        self._emit_storage_metric(reservoir_id, frac)

        # Write output
        ts_str   = timestamp.strftime("%Y%m%d_%H")
        out_path = self.OUTPUT_DIR / f"reservoir_{reservoir_id}_{ts_str}.json"

        result = StorageUpdate(
            reservoir_id      = reservoir_id,
            reservoir_name    = name,
            timestamp         = timestamp.isoformat(),
            storage_m3        = round(new_s, 0),
            active_fraction   = round(frac, 4),
            level_masl        = round(level, 2),
            inflow_cms        = round(inflow_cms, 2),
            outflow_cms       = round(outflow_cms, 2),
            gate_status       = gate_status,
            water_supply_risk = water_supply_risk,
            output_path       = str(out_path),
            notes             = notes,
        )
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(
            "ResStorage | %-15s frac=%.3f level=%.1fmasl Q_in=%.0f Q_out=%.0f status=%s",
            reservoir_id, frac, level, inflow_cms, outflow_cms, gate_status,
        )
        return result

    def compute_flood_gate_release(
        self,
        reservoir_id:         str,
        current_storage_m3:   float,
        inflow_forecast_cms:  list[float],
        timestamp:            datetime | None = None,
    ) -> FloodGateDecision:
        """
        Determine recommended gate release based on rule curve and 6hr inflow forecast.
        Dispatches BPBD alert if gate_status = EMERGENCY or MWL breach imminent.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        cfg       = self._get_cfg(reservoir_id)
        name      = cfg["name"]
        cap       = cfg["active_capacity_m3"]
        dead      = cfg["dead_storage_m3"]
        max_rel   = cfg.get("max_release_cms",    500.0)
        norm_rel  = cfg.get("normal_release_cms", 50.0)
        notes: list[str] = []

        frac      = (current_storage_m3 - dead) / max(cap, 1.0)
        peak_in   = max(inflow_forecast_cms) if inflow_forecast_cms else 0.0
        fcast_hr  = len(inflow_forecast_cms)

        # --- Rule curve decision ---
        if frac < _TOL:
            release       = min(norm_rel * 0.5, max_rel)
            gate_status   = _GATE_NORMAL
            dispatch      = False
            alert_level   = "NONE"
            rationale     = (
                f"Storage at {frac:.0%} (below TOL={_TOL:.0%}). "
                "Minimising spill; priority is water supply."
            )
        elif frac < _FSL:
            release       = min(norm_rel * (1.0 + (frac - _TOL) / (_FSL - _TOL)), max_rel)
            gate_status   = _GATE_NORMAL
            dispatch      = False
            alert_level   = "NONE"
            rationale     = (
                f"Storage at {frac:.0%}; normal operations. "
                f"Release scaled to match downstream demand: {release:.0f} m³/s."
            )
        elif frac < _MWL:
            # Pre-release: target reduction of 10% capacity over 6 hrs
            target_drawdown_m3 = cap * 0.10
            release = min(
                norm_rel + target_drawdown_m3 / (6 * 3600),
                max_rel,
            )
            gate_status = _GATE_ALERT
            dispatch    = True
            alert_level = "FSL_EXCEEDED"
            rationale   = (
                f"Storage at {frac:.0%} (above FSL={_FSL:.0%}). "
                f"Pre-releasing {release:.0f} m³/s to recover FSL headroom. "
                "Downstream communities alerted."
            )
        else:
            release       = max_rel
            gate_status   = _GATE_EMERGENCY
            dispatch      = True
            alert_level   = "MWL_BREACH"
            rationale     = (
                f"EMERGENCY: Storage at {frac:.0%} — MWL exceeded. "
                f"Maximum gate opening {release:.0f} m³/s (max). BPBD immediate dispatch."
            )
            notes.append("MWL breach — emergency spill; BPBD dispatch")

        # Dispatch alert
        if dispatch:
            self._dispatch_reservoir_alert(
                reservoir_id, name, gate_status, alert_level,
                frac, release, peak_in, timestamp, notes,
            )

        out_path = self.OUTPUT_DIR / f"gate_{reservoir_id}_{timestamp.strftime('%Y%m%d_%H')}.json"
        result = FloodGateDecision(
            reservoir_id             = reservoir_id,
            reservoir_name           = name,
            recommended_release_cms  = round(release, 2),
            gate_status              = gate_status,
            rationale                = rationale,
            dispatch_alert           = dispatch,
            alert_level              = alert_level,
            forecast_horizon_hr      = fcast_hr,
            peak_inflow_cms          = round(peak_in, 2),
            output_path              = str(out_path),
            notes                    = notes,
        )
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(
            "GateDecision | %-15s frac=%.3f release=%.0f status=%s alert=%s",
            reservoir_id, frac, release, gate_status, alert_level,
        )
        return result

    def get_downstream_impact(
        self,
        reservoir_id: str,
        release_cms:  float,
    ) -> DownstreamImpact:
        """
        Estimate peak downstream discharge and affected cities given release rate.
        Uses travel time and attenuation factor from reservoir config.
        """
        cfg      = self._get_cfg(reservoir_id)
        travel_h = cfg.get("travel_time_hr", cfg.get("travel_time_karawang_hr", 8.0))
        cities   = cfg.get("downstream_cities", [])
        # Peak attenuation: 60% of release reaches downstream peak (floodplain routing)
        peak_cms = release_cms * 0.60
        # Impact level by fraction of max release
        max_r    = cfg.get("max_release_cms", 500.0)
        ratio    = release_cms / max(max_r, 1.0)
        if ratio >= 0.80:
            level = "CRITICAL"
        elif ratio >= 0.60:
            level = "HIGH"
        elif ratio >= 0.35:
            level = "MODERATE"
        else:
            level = "LOW"

        return DownstreamImpact(
            reservoir_id    = reservoir_id,
            release_cms     = round(release_cms, 2),
            peak_cms_24hr   = round(peak_cms, 2),
            travel_time_hr  = travel_h,
            affected_cities = cities,
            impact_level    = level,
        )

    def get_active_operations(self) -> dict[str, Any]:
        """Return current storage state for all 5 reservoirs (from in-memory cache)."""
        if self.ACTIVE_OPS.exists():
            try:
                with open(self.ACTIVE_OPS) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def update_active_operations(
        self,
        updates:   list[StorageUpdate],
        decisions: list[FloodGateDecision],
        timestamp: datetime,
    ) -> None:
        """Write rolling active_operations.json sidecar for GEOSPATIAL + VISUALIA."""
        gate_order   = {_GATE_EMERGENCY: 2, _GATE_ALERT: 1, _GATE_NORMAL: 0}
        ops          = {}
        dec_map      = {d.reservoir_id: d for d in decisions}

        for u in updates:
            dec = dec_map.get(u.reservoir_id)
            ops[u.reservoir_id] = {
                "name":             u.reservoir_name,
                "gate_status":      u.gate_status,
                "active_fraction":  u.active_fraction,
                "level_masl":       u.level_masl,
                "inflow_cms":       u.inflow_cms,
                "outflow_cms":      u.outflow_cms,
                "recommended_release_cms": dec.recommended_release_cms if dec else u.outflow_cms,
                "alert_level":      dec.alert_level if dec else "NONE",
                "water_supply_risk":u.water_supply_risk,
            }

        worst = max(
            (ops[r]["gate_status"] for r in ops),
            key=lambda s: gate_order.get(s, 0),
            default=_GATE_NORMAL,
        )
        active = {
            "generated_at":  timestamp.isoformat(),
            "reservoir_count": len(ops),
            "highest_gate_status": worst,
            "reservoirs":    ops,
        }
        with open(self.ACTIVE_OPS, "w") as f:
            json.dump(active, f, indent=2)
        logger.info(
            "active_operations.json updated | %d reservoirs | highest=%s",
            len(ops), worst,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_cfg(self, reservoir_id: str) -> dict[str, Any]:
        cfg = self._config.get(reservoir_id)
        if cfg is None:
            raise ValueError(
                f"Unknown reservoir_id '{reservoir_id}'. Supported: {list(self._config.keys())}"
            )
        return cfg

    def _load_config(self) -> dict[str, Any]:
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("reservoir_config.json read error: %s; using defaults", exc)
        # Minimal inline defaults
        return {
            r: {"name": r, "active_capacity_m3": 1e9, "dead_storage_m3": 5e7,
                "FSL_masl": 100.0, "MWL_masl": 102.0, "max_release_cms": 500.0,
                "normal_release_cms": 50.0, "downstream_cities": [], "travel_time_hr": 6.0}
            for r in ALL_RESERVOIRS
        }

    def _dispatch_reservoir_alert(
        self,
        reservoir_id: str,
        name:         str,
        gate_status:  str,
        alert_level:  str,
        frac:         float,
        release:      float,
        peak_in:      float,
        timestamp:    datetime,
        notes:        list[str],
    ) -> None:
        alert = {
            "event_type":      "RESERVOIR_GATE_ALERT",
            "reservoir_id":    reservoir_id,
            "reservoir_name":  name,
            "gate_status":     gate_status,
            "alert_level":     alert_level,
            "storage_fraction":round(frac, 4),
            "release_cms":     round(release, 2),
            "peak_inflow_cms": round(peak_in, 2),
            "issued_at":       timestamp.isoformat(),
        }
        # Downstream impact
        impact = self.get_downstream_impact(reservoir_id, release)
        alert["downstream_impact"] = asdict(impact)

        queue = self.WORKSPACE / "output" / "alerts" / "reservoir_alerts.jsonl"
        queue.parent.mkdir(parents=True, exist_ok=True)
        with open(queue, "a") as f:
            f.write(json.dumps(alert) + "\n")

        bpbd_url = os.environ.get("BPBD_WEBHOOK_URL", "")
        if bpbd_url:
            try:
                req = urllib.request.Request(
                    bpbd_url,
                    data=json.dumps(alert).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10):
                    pass
            except Exception as exc:
                notes.append(f"BPBD webhook error ({exc}); alert queued")
                logger.error("BPBD webhook failed for %s: %s", reservoir_id, exc)

    @staticmethod
    def _emit_storage_metric(reservoir_id: str, frac: float) -> None:
        try:
            from src.hydrology.metrics import RESERVOIR_STORAGE_FRACTION
            RESERVOIR_STORAGE_FRACTION.labels(reservoir_id=reservoir_id).set(frac)
        except Exception as exc:
            logger.debug("RESERVOIR_STORAGE_FRACTION emit error: %s", exc)
