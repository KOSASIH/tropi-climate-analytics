"""Seasonal water availability forecasts for agricultural planning and reservoir management.

Public API (imported by CLOUD-FORGE DAGs: dags/seasonal_monthly.py, VISUALIA dashboard payload):
  SeasonalWaterAvailabilityPipeline — main pipeline class
  WaterAvailabilityClass            — SCARCE/DEFICIT/ADEQUATE/SURPLUS/ABUNDANT enum
  PlantingRecommendation            — Kementan planting advisory enum
  ENSOState                         — ENSO phase enum (La Niña / El Niño / Neutral)
  SeasonalOutlook                   — per-basin 3-month forecast dataclass
  SeasonalForecastStatus            — monthly run result model
  ENSOMonitor                       — ENSO/IOD teleconnection utility
  WAICalculator                     — Water Availability Index composite utility
  PlantingAdvisor                   — zone advisory engine
"""

from src.hydrology.seasonal.water_availability import (
    SeasonalWaterAvailabilityPipeline,
    WaterAvailabilityClass,
    PlantingRecommendation,
    ENSOState,
    SeasonalOutlook,
    SeasonalForecastStatus,
    ENSOMonitor,
    WAICalculator,
    PlantingAdvisor,
)

__all__ = [
    "SeasonalWaterAvailabilityPipeline",
    "WaterAvailabilityClass",
    "PlantingRecommendation",
    "ENSOState",
    "SeasonalOutlook",
    "SeasonalForecastStatus",
    "ENSOMonitor",
    "WAICalculator",
    "PlantingAdvisor",
]
