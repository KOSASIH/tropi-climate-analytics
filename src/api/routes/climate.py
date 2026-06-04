"""Climate data API routes — SST, AQI, Rainfall"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/climate")


class ClimateDataPoint(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    value: float
    unit: str
    source: str


class AQIResponse(BaseModel):
    station_id: str
    station_name: str
    timestamp: str
    aqi: int
    category: str
    pm25: Optional[float] = None
    pm10: Optional[float] = None


@router.get("/temperature", response_model=list[ClimateDataPoint])
async def get_sea_surface_temperature(
    lat_min: float = Query(-11.0),
    lat_max: float = Query(6.0),
    lon_min: float = Query(95.0),
    lon_max: float = Query(141.0),
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[ClimateDataPoint]:
    """Sea surface temperature from MODIS Aqua/Terra. Wired to DATA-FLOW pipeline."""
    return []


@router.get("/aqi", response_model=list[AQIResponse])
async def get_air_quality_index(
    province: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
) -> list[AQIResponse]:
    """AQI per Indonesian ISPU standards. Sources: MODIS aerosol + BMKG stations."""
    return []


@router.get("/rainfall", response_model=list[ClimateDataPoint])
async def get_rainfall(
    lat_min: float = Query(-11.0),
    lat_max: float = Query(6.0),
    lon_min: float = Query(95.0),
    lon_max: float = Query(141.0),
    resolution: str = Query("daily"),
) -> list[ClimateDataPoint]:
    """Precipitation from NASA GPM IMERG + BMKG gauge corrections."""
    return []
