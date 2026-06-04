"""Flood prediction — LSTM-based 6/12/24h streamflow forecasting and alert generation.

Public API (imported by CLOUD-FORGE DAG: dags/flood_early_warning_30min.py):
  FloodEarlyWarningPipeline  — main pipeline class
  AlertAPIClient             — JWT-authenticated alert push/resolve client
  FloodAlertPayload          — Pydantic model for alert ingestion schema
  EarlyWarningRunStatus      — run result model
  FloodAlertLevel            — Siaga 1/2/3 enum
  StreamflowForecast         — per-river LSTM forecast dataclass
"""

from src.hydrology.flood.flood_prediction import (
    FloodEarlyWarningPipeline,
    FloodAlertPayload,
    EarlyWarningRunStatus,
    FloodAlertLevel,
    StreamflowForecast,
    RIVER_CONFIGS,
)
from src.hydrology.flood.alert_client import (
    AlertAPIClient,
    AlertPushResult,
    AlertIngestResponse,
    AlertResolveResponse,
)

__all__ = [
    # Pipeline
    "FloodEarlyWarningPipeline",
    "FloodAlertPayload",
    "EarlyWarningRunStatus",
    "FloodAlertLevel",
    "StreamflowForecast",
    "RIVER_CONFIGS",
    # Auth client
    "AlertAPIClient",
    "AlertPushResult",
    "AlertIngestResponse",
    "AlertResolveResponse",
]
