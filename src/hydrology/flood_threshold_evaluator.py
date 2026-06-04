"""
flood_threshold_evaluator.py — Sprint 6 E5
FloodThresholdEvaluator: Evaluate streamflow forecast against river-specific
thresholds and trigger emergency_flood_alert_dag on breach.

Thresholds (m³/s):
  Ciliwung:  WATCH=150  / WARNING=300  / EMERGENCY=500
  Brantas:   WATCH=800  / WARNING=1500 / EMERGENCY=2500
  Solo:      WATCH=1200 / WARNING=2200 / EMERGENCY=3500

On breach: TriggerDagRunOperator conf={river_id, flood_stage, peak_cms,
           forecast_horizon_hr} → emergency_flood_alert_dag

Prometheus:
  tropi_flood_threshold_breach_total{river_id, flood_stage}  Counter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold catalogue
# ---------------------------------------------------------------------------

_THRESHOLDS: dict[str, dict[str, float]] = {
    "ciliwung": {"WATCH": 150.0,  "WARNING": 300.0,  "EMERGENCY": 500.0},
    "brantas":  {"WATCH": 800.0,  "WARNING": 1500.0, "EMERGENCY": 2500.0},
    "solo":     {"WATCH": 1200.0, "WARNING": 2200.0, "EMERGENCY": 3500.0},
}

# Stage ordering (highest first for evaluation priority)
_STAGE_ORDER = ["EMERGENCY", "WARNING", "WATCH"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThresholdEvalResult:
    river_id:            str
    peak_cms:            float
    forecast_horizon_hr: int
    flood_stage:         str        # "NONE" | "WATCH" | "WARNING" | "EMERGENCY"
    threshold_breached:  float | None   # m³/s value that was breached
    trigger_dispatched:  bool
    trigger_conf:        dict[str, Any] = field(default_factory=dict)
    warnings:            list[str]      = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FloodThresholdEvaluator:
    """
    Evaluate a StreamflowForecastResult against river thresholds and
    optionally trigger emergency_flood_alert_dag via Airflow API.
    """

    EMERGENCY_DAG_ID = "emergency_flood_alert_dag"

    def __init__(
        self,
        airflow_api_url: str | None = None,
        airflow_username: str | None = None,
        airflow_password: str | None = None,
    ) -> None:
        import os
        self._airflow_url = (
            airflow_api_url
            or os.environ.get("AIRFLOW_API_URL", "http://airflow-webserver:8080")
        )
        self._airflow_user = airflow_username or os.environ.get("AIRFLOW_API_USER", "airflow")
        self._airflow_pass = airflow_password or os.environ.get("AIRFLOW_API_PASS", "airflow")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        river_id: str,
        forecast: Any,   # StreamflowForecastResult (avoid circular import)
    ) -> ThresholdEvalResult:
        """
        Evaluate *forecast* against thresholds for *river_id*.

        Iterates through all forecast horizons, picks the worst-case (highest
        peak_cms) horizon, then checks against EMERGENCY → WARNING → WATCH.

        On any breach, calls _trigger_alert_dag() to dispatch
        emergency_flood_alert_dag via Airflow REST API.
        """
        river_id = river_id.lower()
        warnings: list[str] = []

        if river_id not in _THRESHOLDS:
            warnings.append(
                f"No thresholds defined for river '{river_id}'. "
                f"Supported: {list(_THRESHOLDS.keys())}"
            )
            return ThresholdEvalResult(
                river_id=river_id, peak_cms=0.0, forecast_horizon_hr=0,
                flood_stage="NONE", threshold_breached=None,
                trigger_dispatched=False, warnings=warnings,
            )

        thresholds = _THRESHOLDS[river_id]

        # Pick worst-case horizon
        worst_horizon = max(forecast.horizons, key=lambda h: h.peak_cms)
        peak_cms      = worst_horizon.peak_cms
        horizon_hr    = worst_horizon.horizon_hr

        # Classify flood stage
        flood_stage, threshold_breached = self._classify_stage(peak_cms, thresholds)

        trigger_dispatched = False
        trigger_conf: dict[str, Any] = {}

        if flood_stage != "NONE":
            # Emit counter metric
            self._emit_breach_metric(river_id, flood_stage)

            # Build TriggerDagRunOperator conf
            trigger_conf = {
                "river_id":            river_id,
                "flood_stage":         flood_stage,
                "peak_cms":            peak_cms,
                "forecast_horizon_hr": horizon_hr,
            }

            # Dispatch emergency DAG
            trigger_dispatched = self._trigger_alert_dag(trigger_conf)

            logger.warning(
                "Flood threshold BREACHED | river=%s stage=%s peak=%.1f m³/s horizon=%dhr triggered=%s",
                river_id, flood_stage, peak_cms, horizon_hr, trigger_dispatched,
            )
        else:
            logger.info(
                "Flood threshold OK | river=%s peak=%.1f m³/s (below WATCH=%.1f)",
                river_id, peak_cms, thresholds["WATCH"],
            )

        return ThresholdEvalResult(
            river_id            = river_id,
            peak_cms            = peak_cms,
            forecast_horizon_hr = horizon_hr,
            flood_stage         = flood_stage,
            threshold_breached  = threshold_breached,
            trigger_dispatched  = trigger_dispatched,
            trigger_conf        = trigger_conf,
            warnings            = warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_stage(
        peak_cms:   float,
        thresholds: dict[str, float],
    ) -> tuple[str, float | None]:
        """Return (flood_stage, breached_threshold_value)."""
        for stage in _STAGE_ORDER:
            if peak_cms >= thresholds[stage]:
                return stage, thresholds[stage]
        return "NONE", None

    def _trigger_alert_dag(self, conf: dict[str, Any]) -> bool:
        """
        POST to Airflow REST API to trigger emergency_flood_alert_dag.
        Equivalent to TriggerDagRunOperator with the given conf dict.
        Returns True on success, False on failure (non-raising).
        """
        import json
        try:
            import urllib.request
            url     = f"{self._airflow_url}/api/v1/dags/{self.EMERGENCY_DAG_ID}/dagRuns"
            payload = json.dumps({"conf": conf}).encode()
            req     = urllib.request.Request(
                url,
                data    = payload,
                headers = {
                    "Content-Type": "application/json",
                    "Accept":       "application/json",
                },
                method  = "POST",
            )
            # Basic auth
            import base64
            creds   = base64.b64encode(
                f"{self._airflow_user}:{self._airflow_pass}".encode()
            ).decode()
            req.add_header("Authorization", f"Basic {creds}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                status_code = resp.status
                if status_code in (200, 201):
                    logger.info(
                        "DAG %s triggered via API | conf=%s",
                        self.EMERGENCY_DAG_ID, conf,
                    )
                    return True
                logger.error(
                    "DAG trigger returned HTTP %d for conf=%s", status_code, conf
                )
                return False

        except Exception as exc:
            logger.error(
                "Failed to trigger %s: %s | conf=%s",
                self.EMERGENCY_DAG_ID, exc, conf,
            )
            return False

    @staticmethod
    def _emit_breach_metric(river_id: str, flood_stage: str) -> None:
        try:
            from src.hydrology.metrics import FLOOD_THRESHOLD_BREACH
            FLOOD_THRESHOLD_BREACH.labels(
                river_id=river_id,
                flood_stage=flood_stage,
            ).inc()
        except Exception as exc:  # pragma: no cover
            logger.debug("Prometheus metrics unavailable: %s", exc)
