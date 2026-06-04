"""Groundwater monitoring — GRACE-FO TWS anomaly, aquifer depletion trend analysis.

Public API (imported by CLOUD-FORGE DAG: dags/grace_monthly.py):
  GRACEFOGroundwaterPipeline — canonical DAG-facing alias for GRACEFOGroundwaterMonitor
  GRACEFOGroundwaterMonitor  — original class name (backward compat)
  AquiferDepletionLevel      — NORMAL/WATCH/WARNING/CRITICAL/EMERGENCY enum
  GroundwaterStorage         — per-aquifer GWS dataclass
  AquiferStatus              — risk + trend summary model
  GRACERunStatus             — monthly run result model
  TrendAnalyzer              — depletion trend utility
"""

from src.hydrology.groundwater.grace_fo_monitor import (
    GRACEFOGroundwaterMonitor,
    AquiferDepletionLevel,
    GroundwaterStorage,
    AquiferStatus,
    GRACERunStatus,
    TrendAnalyzer,
    GRACEGranule,
)

# Canonical alias expected by CLOUD-FORGE DAG imports
GRACEFOGroundwaterPipeline = GRACEFOGroundwaterMonitor

__all__ = [
    # DAG-facing alias (primary export)
    "GRACEFOGroundwaterPipeline",
    # Original class (backward compat)
    "GRACEFOGroundwaterMonitor",
    # Supporting types
    "AquiferDepletionLevel",
    "GroundwaterStorage",
    "AquiferStatus",
    "GRACERunStatus",
    "TrendAnalyzer",
    "GRACEGranule",
]
