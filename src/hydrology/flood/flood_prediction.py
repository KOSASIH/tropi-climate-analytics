"""
Flood Early Warning System — 6/12/24h Lead Time
Agent: HYDROLOGIS | Tropi Climate Analytics

Targeted rivers: Ciliwung (Jakarta), Brantas (Surabaya), Solo (Central Java)

Pipeline:
  1. Ingest QPE 4km/30min from GPM-BMKG fusion pipeline
  2. Ingest SMAP antecedent soil moisture
  3. Run SWAT daily / sub-daily forward simulation
  4. LSTM-based streamflow forecasting (6h, 12h, 24h lead times)
  5. Stage-threshold classification (BNPB Siaga 1/2/3)
  6. Push AlertType.FLOOD to /api/v1/alerts/active via JWT-authenticated AlertAPIClient
     Resolve via /api/v1/alerts/{id}/resolve when stage drops below threshold

Auth: JWT HS256  sub=hydrologis-pipeline  role=internal
      Token cached with TTL < exp (auto-refresh 60s before expiry)
      Config: TROPI_API_BASE_URL, JWT_SECRET (env vars — never hardcode)

Alert levels (BNPB/BPBD standard):
  Siaga 3 (Warning):   Yellow  — Prepare monitoring
  Siaga 2 (Alert):     Orange  — Evacuate at-risk zones
  Siaga 1 (Emergency): Red     — Immediate evacuation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

import numpy as np
from loguru import logger
from pydantic import BaseModel

from src.api.routes.alerts import Alert, AlertType, AlertSeverity
from src.hydrology.flood.alert_client import AlertAPIClient, AlertPushResult
from src.hydrology.swat.ciliwung_model import (
    CiliwungSWATModel,
    CILIWUNG_OUTLET_LAT,
    CILIWUNG_OUTLET_LON,
    FLOOD_STAGE_SIAGA1,
    FLOOD_STAGE_SIAGA2,
    FLOOD_STAGE_SIAGA3,
)


# ---------------------------------------------------------------------------
# River gauge configurations
# ---------------------------------------------------------------------------

RIVER_CONFIGS = {
    "ciliwung": {
        "name": "Ciliwung River — Manggarai",
        "lat": CILIWUNG_OUTLET_LAT,
        "lon": CILIWUNG_OUTLET_LON,
        "province": "DKI Jakarta",
        "area_km2": 387,
        "stage_siaga3": FLOOD_STAGE_SIAGA3,  # 750 cm
        "stage_siaga2": FLOOD_STAGE_SIAGA2,  # 850 cm
        "stage_siaga1": FLOOD_STAGE_SIAGA1,  # 950 cm
        "bankfull_q_m3s": 250.0,
        "affected_population": 1_200_000,
    },
    "brantas": {
        "name": "Brantas River — Mojokerto",
        "lat": -7.47,
        "lon": 112.43,
        "province": "Jawa Timur",
        "area_km2": 11_800,
        "stage_siaga3": 600,
        "stage_siaga2": 750,
        "stage_siaga1": 900,
        "bankfull_q_m3s": 1800.0,
        "affected_population": 780_000,
    },
    "solo": {
        "name": "Solo River (Bengawan Solo) — Bojonegoro",
        "lat": -7.15,
        "lon": 111.88,
        "province": "Jawa Tengah",
        "area_km2": 16_100,
        "stage_siaga3": 700,
        "stage_siaga2": 850,
        "stage_siaga1": 1000,
        "bankfull_q_m3s": 2500.0,
        "affected_population": 1_400_000,
    },
}

# Alert severity mapping from Siaga level
_SIAGA_TO_SEVERITY = {
    "siaga1": AlertSeverity.EMERGENCY,
    "siaga2": AlertSeverity.CRITICAL,
    "siaga3": AlertSeverity.WARNING,
}
_SIAGA_TO_TITLE = {
    "siaga1": "🔴 Flood Emergency",
    "siaga2": "🟠 Flood Alert",
    "siaga3": "🟡 Flood Warning",
}

# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------

class FloodAlertLevel(str, Enum):
    NONE    = "none"
    SIAGA3  = "siaga3"   # Warning (Yellow)
    SIAGA2  = "siaga2"   # Alert (Orange)
    SIAGA1  = "siaga1"   # Emergency (Red)


@dataclass
class StreamflowForecast:
    """LSTM streamflow forecast for one river at multiple lead times."""
    river_id: str
    issued_at: datetime
    current_discharge_m3s: float
    current_stage_cm: float
    forecast_6h_m3s: float
    forecast_12h_m3s: float
    forecast_24h_m3s: float
    stage_6h_cm: float
    stage_12h_cm: float
    stage_24h_cm: float
    alert_level_6h: FloodAlertLevel
    alert_level_12h: FloodAlertLevel
    alert_level_24h: FloodAlertLevel
    peak_alert_level: FloodAlertLevel
    confidence_pct: float
    antecedent_sm_m3m3: float


class FloodAlertPayload(BaseModel):
    """Alert payload compatible with AlertIngestion schema."""
    alert_id: str
    alert_type: AlertType = AlertType.FLOOD
    severity: AlertSeverity
    title: str
    description: str
    affected_area: str
    latitude: float
    longitude: float
    radius_km: float
    issued_at: datetime
    expires_at: Optional[datetime] = None
    source_agent: str = "HYDROLOGIS"
    metadata: Optional[dict] = None


class EarlyWarningRunStatus(BaseModel):
    run_time: datetime
    rivers_assessed: int
    alerts_generated: int
    alerts_pushed: int
    push_failures: int
    most_severe_river: Optional[str] = None
    most_severe_level: Optional[str] = None
    source_agent: str = "HYDROLOGIS"


# ---------------------------------------------------------------------------
# LSTM surrogate model
# ---------------------------------------------------------------------------

class LSTMStreamflowModel:
    """
    LSTM surrogate for streamflow forecasting.
    Production: load TorchScript weights from model registry.
    Sprint 1: parametric surrogate calibrated to basin characteristics.
    """

    def __init__(self, river_id: str) -> None:
        self.river_id = river_id
        cfg = RIVER_CONFIGS[river_id]
        self.bankfull = cfg["bankfull_q_m3s"]

    def predict(
        self,
        current_q: float,
        precip_mm_6h: float,
        sm_m3m3: float,
    ) -> tuple[float, float, float]:
        """
        Returns (q_6h, q_12h, q_24h) in m³/s.
        Simplified unit-hydrograph approximation.
        """
        # Catchment response factor (sm raises sensitivity)
        sm_factor = 1.0 + max(0.0, (sm_m3m3 - 0.25) / 0.15)
        # Rainfall-runoff contribution
        rr_6h  = precip_mm_6h * sm_factor * self.bankfull / 180.0
        rr_12h = rr_6h  * 0.65
        rr_24h = rr_12h * 0.50
        # Recession from current level
        q_6h  = current_q * 0.92 + rr_6h
        q_12h = current_q * 0.82 + rr_12h
        q_24h = current_q * 0.68 + rr_24h
        return q_6h, q_12h, q_24h


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class FloodEarlyWarningPipeline:
    """
    Flood early warning pipeline — 30-minute cadence.

    Consumes:
      - QPE grid from GPMBMKGFusionPipeline (precip forcing)
      - SMAP soil moisture from SMAPSoilMoisturePipeline (antecedent state)
      - SWAT Ciliwung model (streamflow routing for Ciliwung sub-basin)

    Produces:
      - StreamflowForecast per river (6/12/24h)
      - FloodAlertPayload per triggered river
      - Authenticated POST to /api/v1/alerts/active via AlertAPIClient
      - Resolve call to /api/v1/alerts/{id}/resolve when stage clears
    """

    def __init__(
        self,
        rivers: list[str] = None,
        api_base_url: str = None,
    ) -> None:
        self.rivers = rivers or list(RIVER_CONFIGS.keys())
        self._lstm_models = {r: LSTMStreamflowModel(r) for r in self.rivers}
        self._swat = CiliwungSWATModel()  # Ciliwung-specific routing
        self._alert_client = AlertAPIClient(
            base_url=api_base_url or os.getenv("TROPI_API_BASE_URL", "http://localhost:8000")
        )
        # Track active alert IDs per river for deduplication + resolve
        self._active_alerts: dict[str, str] = {}  # river_id → alert_id
        logger.info(f"FloodEarlyWarningPipeline ready | rivers: {self.rivers}")

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def discharge_to_stage(q_m3s: float, river_id: str) -> float:
        """Rating curve approximation per river."""
        import math
        cfg = RIVER_CONFIGS[river_id]
        bf = cfg["bankfull_q_m3s"]
        # Power-law: h = a * Q^b — coefficients tuned per basin
        a = cfg["stage_siaga3"] / (bf ** 0.45)
        return max(a * (max(q_m3s, 0.1) ** 0.45), 100.0)

    @staticmethod
    def classify_alert(stage_cm: float, river_id: str) -> FloodAlertLevel:
        cfg = RIVER_CONFIGS[river_id]
        if stage_cm >= cfg["stage_siaga1"]:
            return FloodAlertLevel.SIAGA1
        elif stage_cm >= cfg["stage_siaga2"]:
            return FloodAlertLevel.SIAGA2
        elif stage_cm >= cfg["stage_siaga3"]:
            return FloodAlertLevel.SIAGA3
        return FloodAlertLevel.NONE

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        precip_mm_6h: float = 0.0,
        sm_m3m3: float = 0.30,
        current_discharge: dict = None,
    ) -> tuple[list[StreamflowForecast], EarlyWarningRunStatus]:
        """
        Execute one 30-min early warning cycle.

        Args:
            precip_mm_6h:      6-hour accumulated precipitation (mm)
            sm_m3m3:           Domain-mean antecedent soil moisture
            current_discharge: {river_id: q_m3s} — current observed discharge
        """
        run_time = datetime.now(timezone.utc)
        q_obs = current_discharge or {r: RIVER_CONFIGS[r]["bankfull_q_m3s"] * 0.3 for r in self.rivers}

        forecasts: list[StreamflowForecast] = []
        alert_payloads: list[FloodAlertPayload] = []
        resolves: list[tuple[str, str]] = []  # (alert_id, reason)

        for river_id in self.rivers:
            cfg = RIVER_CONFIGS[river_id]
            q_now = q_obs.get(river_id, cfg["bankfull_q_m3s"] * 0.3)

            q_6h, q_12h, q_24h = self._lstm_models[river_id].predict(
                current_q=q_now,
                precip_mm_6h=precip_mm_6h,
                sm_m3m3=sm_m3m3,
            )

            s_now  = self.discharge_to_stage(q_now,  river_id)
            s_6h   = self.discharge_to_stage(q_6h,   river_id)
            s_12h  = self.discharge_to_stage(q_12h,  river_id)
            s_24h  = self.discharge_to_stage(q_24h,  river_id)

            al_6h  = self.classify_alert(s_6h,  river_id)
            al_12h = self.classify_alert(s_12h, river_id)
            al_24h = self.classify_alert(s_24h, river_id)
            peak_al = max([al_6h, al_12h, al_24h],
                          key=lambda x: ["none", "siaga3", "siaga2", "siaga1"].index(x.value))

            confidence = max(60.0, 90.0 - abs(sm_m3m3 - 0.30) * 50)

            forecasts.append(StreamflowForecast(
                river_id=river_id,
                issued_at=run_time,
                current_discharge_m3s=round(q_now, 2),
                current_stage_cm=round(s_now, 1),
                forecast_6h_m3s=round(q_6h, 2),
                forecast_12h_m3s=round(q_12h, 2),
                forecast_24h_m3s=round(q_24h, 2),
                stage_6h_cm=round(s_6h, 1),
                stage_12h_cm=round(s_12h, 1),
                stage_24h_cm=round(s_24h, 1),
                alert_level_6h=al_6h,
                alert_level_12h=al_12h,
                alert_level_24h=al_24h,
                peak_alert_level=peak_al,
                confidence_pct=round(confidence, 1),
                antecedent_sm_m3m3=round(sm_m3m3, 3),
            ))

            if peak_al != FloodAlertLevel.NONE:
                siaga = peak_al.value  # e.g. "siaga1"
                date_tag = run_time.strftime("%Y%m%d")
                alert_id = f"flood-{river_id}-{date_tag}-{siaga}"

                # Deduplication: skip if same alert_id already active
                if self._active_alerts.get(river_id) != alert_id:
                    self._active_alerts[river_id] = alert_id
                    alert_payloads.append(FloodAlertPayload(
                        alert_id=alert_id,
                        severity=_SIAGA_TO_SEVERITY[siaga],
                        title=f"{_SIAGA_TO_TITLE[siaga]} — {cfg['name']}",
                        description=(
                            f"Streamflow forecast: {q_6h:.0f}m³/s at 6h, "
                            f"{q_12h:.0f}m³/s at 12h, {q_24h:.0f}m³/s at 24h. "
                            f"Projected stage: {s_6h:.0f}cm. "
                            f"Estimated affected population: {cfg['affected_population']:,}."
                        ),
                        affected_area=cfg["province"],
                        latitude=cfg["lat"],
                        longitude=cfg["lon"],
                        radius_km=float(max(15.0, (cfg["area_km2"] ** 0.5) * 0.5)),
                        issued_at=run_time,
                        expires_at=run_time + timedelta(hours=24),
                        metadata={
                            "river_id": river_id,
                            "current_discharge_m3s": q_now,
                            "forecast_6h_m3s": round(q_6h, 2),
                            "forecast_12h_m3s": round(q_12h, 2),
                            "forecast_24h_m3s": round(q_24h, 2),
                            "stage_6h_cm": round(s_6h, 1),
                            "siaga_level": siaga,
                            "confidence_pct": confidence,
                            "antecedent_sm_m3m3": sm_m3m3,
                        },
                    ))
                    logger.warning(
                        f"FLOOD [{siaga.upper()}] {cfg['name']} | "
                        f"Q={q_now:.0f}→{q_6h:.0f}m³/s | stage={s_6h:.0f}cm"
                    )
            else:
                # Stage cleared — resolve any active alert for this river
                prev_alert = self._active_alerts.pop(river_id, None)
                if prev_alert:
                    resolves.append((prev_alert, "River stage returned to normal level"))

        # Push new alerts
        push_results = self.push_to_alert_api(alert_payloads)

        # Resolve cleared alerts
        for alert_id, reason in resolves:
            self.resolve_alert(alert_id, reason)

        n_pushed = sum(1 for r in push_results if r.success)
        n_failed = len(push_results) - n_pushed

        most_severe = max(
            [f for f in forecasts if f.peak_alert_level != FloodAlertLevel.NONE],
            key=lambda f: ["none", "siaga3", "siaga2", "siaga1"].index(f.peak_alert_level.value),
            default=None,
        )

        status = EarlyWarningRunStatus(
            run_time=run_time,
            rivers_assessed=len(self.rivers),
            alerts_generated=len(alert_payloads),
            alerts_pushed=n_pushed,
            push_failures=n_failed,
            most_severe_river=most_severe.river_id if most_severe else None,
            most_severe_level=most_severe.peak_alert_level.value if most_severe else None,
        )
        logger.info(
            f"EarlyWarning run: {len(self.rivers)} rivers | "
            f"alerts={len(alert_payloads)} pushed={n_pushed} failed={n_failed}"
        )
        return forecasts, status

    # ------------------------------------------------------------------
    # Alert push — JWT authenticated via AlertAPIClient
    # ------------------------------------------------------------------

    def push_to_alert_api(
        self,
        alerts: list[FloodAlertPayload],
    ) -> list[AlertPushResult]:
        """
        POST flood alerts to POST /api/v1/alerts/active.

        Auth: JWT HS256  sub=hydrologis-pipeline  role=internal
        Token cached (TTL < exp, auto-refresh 60s before expiry).
        Config: TROPI_API_BASE_URL, JWT_SECRET (env vars).
        """
        if not alerts:
            return []
        payloads = [a.model_dump(mode="json") for a in alerts]
        results = self._alert_client.push_alerts(payloads)
        for r in results:
            if not r.success:
                logger.error(
                    f"Failed to push alert {r.alert_id} "
                    f"(HTTP {r.status_code}): {r.message}"
                )
        return results

    def resolve_alert(
        self,
        alert_id: str,
        reason: str = "River level dropped below flood threshold",
    ) -> None:
        """
        POST /api/v1/alerts/{alert_id}/resolve when river stage clears.
        Auth: same JWT token (cached, role=internal).
        """
        result = self._alert_client.resolve_alert(alert_id, reason)
        if result.resolved:
            logger.info(f"Resolved: {alert_id}")
        else:
            logger.warning(f"Resolve failed for {alert_id}: {result.reason}")
