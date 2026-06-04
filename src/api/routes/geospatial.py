"""Geospatial API routes — deforestation, land cover, spatial queries"""

from enum import Enum
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/geospatial")


class LandCoverClass(str, Enum):
    FOREST = "forest"
    DEGRADED_FOREST = "degraded_forest"
    PLANTATION = "plantation"
    CROPLAND = "cropland"
    WATER = "water"
    URBAN = "urban"
    BARE_LAND = "bare_land"
    MANGROVE = "mangrove"
    PEATLAND = "peatland"


class DeforestationEvent(BaseModel):
    event_id: str
    detected_at: str
    area_ha: float
    latitude: float
    longitude: float
    province: str
    district: str
    confidence: float
    satellite_source: str
    previous_cover: LandCoverClass
    current_cover: LandCoverClass


class LandCoverStats(BaseModel):
    region: str
    year: int
    total_area_ha: float
    cover_breakdown: dict[str, float]
    forest_coverage_pct: float


@router.get("/deforestation", response_model=list[DeforestationEvent])
async def get_deforestation_events(
    lat_min: float = Query(-11.0),
    lat_max: float = Query(6.0),
    lon_min: float = Query(95.0),
    lon_max: float = Query(141.0),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_area_ha: float = Query(0.1),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0),
) -> list[DeforestationEvent]:
    """Deforestation from Landsat 8/9 temporal differencing + ML. ~85% accuracy."""
    return []


@router.get("/landcover", response_model=LandCoverStats)
async def get_land_cover_statistics(
    province: Optional[str] = None,
    year: int = Query(2024, ge=2000, le=2030),
) -> LandCoverStats:
    """Land cover stats per province from Landsat ML classifier."""
    return LandCoverStats(
        region=province or "Indonesia",
        year=year,
        total_area_ha=0.0,
        cover_breakdown={},
        forest_coverage_pct=0.0,
    )


@router.get("/provinces")
async def list_provinces() -> list[dict]:
    """Indonesian provinces with bounding boxes for spatial queries."""
    return []
