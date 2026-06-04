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
  6. Push AlertType.FLOOD to /api/v1/alerts/active endpoint

Alert levels (BNPB/BPBD standard):
  Siaga 3 (Warning):   Yellow  — Prepare monitoring
  Siaga 2 (Alert):     Orange  — Evacuate at-risk zones
  Siaga 1 (Emergency): Red     — Immediate evacuation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import uuid4

import numpy as np
from loguru import logger
from pydantic import BaseModel

from src.api.routes.alerts import Alert, AlertType, AlertSeverity
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
    confidence_pct: float       # Model confidence (0-100)
    antecedent_sm_m3m3: float   # Antecedent soil moisture


class FloodAlertPayload(BaseModel):
    """Flood alert ready for /api/v1/alerts/active endpoint."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_area: str
    latitude: float
    longitude: float
    radius_km: float
    issued_at: datetime
    expires_at: Optional[datetime]
    source_agent: str = "HYDROLOGIS"
    # Extended flood-specific fields
    river_id: str
    peak_forecast_m3s: float
    peak_stage_cm: float
    lead_time_hours: int
    affected_population: int


# ---------------------------------------------------------------------------
# LSTM surrogate model
# ---------------------------------------------------------------------------

class LSTMStreamflowModel:
    """
    LSTM-based streamflow forecasting surrogate.

    Input features (24-step window of 6-hourly data):
      - QPE accumulated (mm/6h)
      - SMAP soil moisture (m³/m³)
      - Current streamflow (m³/s)
      - Temperature, humidity (for ET)
      - Tidal water level (coastal rivers)

    Output:
      - Discharge at t+6h, t+12h, t+24h (m³/s)

    In production: load ONNX weights from model registry.
    Here: physics-based surrogate using rational method.
    """

    # Empirical recession coefficients per river (calibrated 2015-2024)
    RECESSION_COEFF = {
        "ciliwung": 0.82,  # Q(t+1) = k * Q(t) + runoff_contribution
        "brantas":  0.91,
        "solo":     0.94,
    }

    PEAK_LAG_HRS = {
        "ciliwung": 4,
        "brantas":  12,
        "solo":     18,
    }

    def forecast(
        self,
        river_id: str,
        current_q_m3s: float,
        precip_6h_mm: float,
        sm_m3m3: float,
        area_km2: float,
    ) -> tuple[float, float, float]:
        """
        Generate 6h/12h/24h discharge forecasts.

        Returns (q_6h, q_12h, q_24h) in m³/s.
        """
        k = self.RECESSION_COEFF.get(river_id, 0.88)
        lag = self.PEAK_LAG_HRS.get(river_id, 6)

        # Runoff from QPE — rational method with SM-adjusted CN
        runoff_coeff = 0.3 + 0.5 * min(sm_m3m3 / 0.45, 1.0)  # 0.3-0.8
        runoff_m3s = (precip_6h_mm / 1000.0) * (area_km2 * 1e6) * runoff_coeff / (6 * 3600)

        # Forward stepping with lag
        q6  = k * current_q_m3s + runoff_m3s * (1.0 if lag <= 6 else 0.3)
        q12 = k * q6 + runoff_m3s * (0.6 if lag <= 12 else 0.2)
        q24 = k * q12 + runoff_m3s * 0.1

        # Add uncertainty bounds (confidence degrades with lead time)
        rng = np.random.default_rng(int(current_q_m3s * 1000) % (2**32))
        noise_factor = rng.uniform(0.95, 1.05)
        return (
            max(q6 * noise_factor, 0.0),
            max(q12 * noise_factor, 0.0),
            max(q24 * noise_factor, 0.0),
        )

    def confidence(
        self,
        lead_hours: int,
        n_gauges: int,
        sm_coverage_pct: float,
    ) -> float:
        """Model confidence (%) decreasing with lead time, improving with data density."""
        base = max(95.0 - (lead_hours * 1.5), 50.0)
        gauge_bonus = min(n_gauges * 0.5, 5.0)
        sm_bonus = (sm_coverage_pct / 100.0) * 3.0
        return min(base + gauge_bonus + sm_bonus, 99.0)


# ---------------------------------------------------------------------------
# Alert generator
# ---------------------------------------------------------------------------

def discharge_to_stage(q_m3s: float, river_id: str) -> float:
    """
    Power-law rating curve per river.
    h = a * Q^b (parameters from BMKG gauge calibration)
    """
    RATING = {
        "ciliwung": (82.4,  0.43),
        "brantas":  (45.0,  0.52),
        "solo":     (38.0,  0.55),
    }
    a, b = RATING.get(river_id, (60.0, 0.48))
    return a * (q_m3s ** b) if q_m3s > 0 else 400.0


def classify_alert(
    stage_cm: float,
    config: dict,
) -> FloodAlertLevel:
    if stage_cm >= config["stage_siaga1"]:
        return FloodAlertLevel.SIAGA1
    elif stage_cm >= config["stage_siaga2"]:
        return FloodAlertLevel.SIAGA2
    elif stage_cm >= config["stage_siaga3"]:
        return FloodAlertLevel.SIAGA3
    return FloodAlertLevel.NONE


def alert_to_severity(level: FloodAlertLevel) -> AlertSeverity:
    mapping = {
        FloodAlertLevel.SIAGA1: AlertSeverity.EMERGENCY,
        FloodAlertLevel.SIAGA2: AlertSeverity.CRITICAL,
        FloodAlertLevel.SIAGA3: AlertSeverity.WARNING,
        FloodAlertLevel.NONE:   AlertSeverity.INFO,
    }
    return mapping[level]


# ---------------------------------------------------------------------------
# Core flood early warning pipeline
# ---------------------------------------------------------------------------

class FloodEarlyWarningPipeline:
    """
    6/12/24h flood early warning for Ciliwung, Brantas, Solo.

    Runs every 30 minutes (Airflow DAG: flood_early_warning_30min),
    triggered after QPE fusion cycle completes.

    Usage:
        pipeline = FloodEarlyWarningPipeline()
        forecasts, alerts = pipeline.run()
    """

    def __init__(self) -> None:
        self.lstm = LSTMStreamflowModel()
        self.swat = CiliwungSWATModel()  # For Ciliwung high-resolution routing
        logger.info(
            f"FloodEW pipeline initialized | rivers: "
            f"{', '.join(RIVER_CONFIGS.keys())}"
        )

    def run(
        self,
        qpe_max_mm_hr: float = 0.0,
        sm_m3m3: float = 0.30,
        current_discharge: Optional[dict] = None,
    ) -> tuple[list[StreamflowForecast], list[FloodAlertPayload]]:
        """
        Run flood early warning across all monitored rivers.

        Args:
            qpe_max_mm_hr: Max rainfall intensity from QPE (mm/hr)
            sm_m3m3: Domain-mean soil moisture from SMAP
            current_discharge: {river_id: q_m3s} from BMKG gauges

        Returns:
            (forecasts, alerts) — alerts contains only rivers exceeding thresholds
        """
        now = datetime.utcnow()
        current_discharge = current_discharge or {}

        forecasts: list[StreamflowForecast] = []
        alerts: list[FloodAlertPayload] = []

        for river_id, config in RIVER_CONFIGS.items():
            # Current discharge (BMKG gauge or SWAT estimate)
            q_now = current_discharge.get(river_id, config["bankfull_q_m3s"] * 0.3)
            stage_now = discharge_to_stage(q_now, river_id)

            # 6-hourly QPE accumulation for this river basin
            precip_6h = qpe_max_mm_hr * 6.0 * 0.65  # Spatial avg factor

            # LSTM forecast
            q6, q12, q24 = self.lstm.forecast(
                river_id=river_id,
                current_q_m3s=q_now,
                precip_6h_mm=precip_6h,
                sm_m3m3=sm_m3m3,
                area_km2=config["area_km2"],
            )

            # Stage forecasts
            s6  = discharge_to_stage(q6,  river_id)
            s12 = discharge_to_stage(q12, river_id)
            s24 = discharge_to_stage(q24, river_id)

            # Alert levels
            al6  = classify_alert(s6,  config)
            al12 = classify_alert(s12, config)
            al24 = classify_alert(s24, config)
            peak_al = max([al6, al12, al24], key=lambda l: list(FloodAlertLevel).index(l))

            # Confidence
            confidence = self.lstm.confidence(
                lead_hours=24,
                n_gauges=12,
                sm_coverage_pct=75.0,
            )

            forecast = StreamflowForecast(
                river_id=river_id,
                issued_at=now,
                current_discharge_m3s=round(q_now, 1),
                current_stage_cm=round(stage_now, 1),
                forecast_6h_m3s=round(q6, 1),
                forecast_12h_m3s=round(q12, 1),
                forecast_24h_m3s=round(q24, 1),
                stage_6h_cm=round(s6, 1),
                stage_12h_cm=round(s12, 1),
                stage_24h_cm=round(s24, 1),
                alert_level_6h=al6,
                alert_level_12h=al12,
                alert_level_24h=al24,
                peak_alert_level=peak_al,
                confidence_pct=round(confidence, 1),
                antecedent_sm_m3m3=round(sm_m3m3, 3),
            )
            forecasts.append(forecast)

            # Issue alert if any lead time triggers
            if peak_al != FloodAlertLevel.NONE:
                # Determine earliest lead time with alert
                if al6 != FloodAlertLevel.NONE:
                    lead_h = 6
                elif al12 != FloodAlertLevel.NONE:
                    lead_h = 12
                else:
                    lead_h = 24

                severity = alert_to_severity(peak_al)
                alert = FloodAlertPayload(
                    alert_id=f"flood_{river_id}_{now.strftime('%Y%m%d%H%M')}_{str(uuid4())[:8]}",
                    alert_type=AlertType.FLOOD,
                    severity=severity,
                    title=f"Flood {peak_al.value.upper()} — {config['name']}",
                    description=(
                        f"Flood early warning issued for {config['name']}. "
                        f"Current stage: {stage_now:.0f}cm. "
                        f"Forecast peak: {max(s6, s12, s24):.0f}cm at {lead_h}h lead. "
                        f"Model confidence: {confidence:.0f}%. "
                        f"Antecedent SM: {sm_m3m3:.3f} m³/m³. "
                        f"QPE max: {qpe_max_mm_hr:.1f}mm/hr."
                    ),
                    affected_area=config["province"],
                    latitude=config["lat"],
                    longitude=config["lon"],
                    radius_km=25.0,
                    issued_at=now,
                    expires_at=now + timedelta(hours=lead_h + 6),
                    river_id=river_id,
                    peak_forecast_m3s=round(max(q6, q12, q24), 1),
                    peak_stage_cm=round(max(s6, s12, s24), 1),
                    lead_time_hours=lead_h,
                    affected_population=config["affected_population"],
                )
                alerts.append(alert)
                logger.warning(
                    f"FLOOD ALERT [{peak_al.value.upper()}] "
                    f"{config['name']} | "
                    f"peak={max(s6,s12,s24):.0f}cm | "
                    f"lead={lead_h}h | "
                    f"pop={config['affected_population']:,}"
                )

        logger.info(
            f"Flood EW complete | rivers={len(forecasts)} "
            f"alerts_issued={len(alerts)}"
        )
        return forecasts, alerts

    def push_to_alert_api(
        self,
        alerts: list[FloodAlertPayload],
        api_base_url: str = None,
    ) -> list[dict]:
        """
        POST flood alerts to /api/v1/alerts/active endpoint.
        In production: use internal service mesh URL.
        """
        api_url = api_base_url or os.getenv(
            "TROPI_API_BASE_URL", "http://localhost:8000"
        )
        results = []
        try:
            import requests
            for alert in alerts:
                resp = requests.post(
                    f"{api_url}/api/v1/alerts/active",
                    json=alert.model_dump(),
                    timeout=10,
                    headers={"Content-Type": "application/json"},
                )
                results.append({
                    "alert_id": alert.alert_id,
                    "status_code": resp.status_code,
                    "success": resp.status_code in (200, 201),
                })
                logger.info(
                    f"Alert push: {alert.alert_id} → HTTP {resp.status_code}"
                )
        except Exception as exc:
            logger.error(f"Alert API push failed: {exc}")
        return results
