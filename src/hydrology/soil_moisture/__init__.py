"""Soil moisture processing — SMAP L3 ingestion, anomaly detection, drought risk index.

Public API (imported by CLOUD-FORGE DAG: dags/smap_daily.py):
  SMAPIngestionPipeline    — canonical DAG-facing alias for SMAPSoilMoisturePipeline
  SMAPSoilMoisturePipeline — original class name (backward compat)
  DroughtRiskLevel         — 0–4 drought severity enum
  SMAPObservation          — single 9km-grid observation dataclass
  SMAAnomalyResult         — SMA anomaly + risk level per cell
  SMAPRunStatus            — daily run result model
"""

from src.hydrology.soil_moisture.smap_pipeline import (
    SMAPSoilMoisturePipeline,
    DroughtRiskLevel,
    SMAPObservation,
    SMAAnomalyResult,
    SMAPRunStatus,
    SMAPClimatology,
)

# Canonical alias expected by CLOUD-FORGE DAG imports
SMAPIngestionPipeline = SMAPSoilMoisturePipeline

__all__ = [
    # DAG-facing alias (primary export)
    "SMAPIngestionPipeline",
    # Original class (backward compat)
    "SMAPSoilMoisturePipeline",
    # Supporting types
    "DroughtRiskLevel",
    "SMAPObservation",
    "SMAAnomalyResult",
    "SMAPRunStatus",
    "SMAPClimatology",
]
