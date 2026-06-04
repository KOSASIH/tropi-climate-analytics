"""Pydantic v2 models for Tropi Climate Analytics API.

Covers all request/response types across:
    - Climate current conditions
    - Forecasting
    - Flood alerts
    - Streamflow
    - Satellite imagery
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UserRole(str, Enum):
    PUBLIC = "public"
    RESEARCHER = "researcher"
    GOVERNMENT = "government"
    ADMIN = "admin"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    FLOOD = "flood"
    DROUGHT = "drought"
    FIRE = "fire"
    AQI = "aqi"
    DEFORESTATION = "deforestation"
    CYCLONE = "cyclone"
    SST_ANOMALY = "sst_anomaly"


class SatelliteMission(str, Enum):
    LANDSAT_8 = "LANDSAT_8"
    LANDSAT_9 = "LANDSAT_9"
    MODIS_TERRA = "MODIS_TERRA"
    MODIS_AQUA = "MODIS_AQUA"
    SMAP = "SMAP"
    GPM_IMERG = "GPM_IMERG"
    SENTINEL_2 = "SENTINEL_2"
    VIIRS_SUOMI = "VIIRS_SUOMI"


class StreamflowStatus(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    FLOOD_WATCH = "flood_watch"
    FLOOD_WARNING = "flood_warning"
    MAJOR_FLOOD = "major_flood"


# ---------------------------------------------------------------------------
# Shared / primitive schemas
# ---------------------------------------------------------------------------


class Coordinate(BaseModel):
    """WGS-84 geographic coordinate."""

    latitude: Annotated[float, Field(ge=-90.0, le=90.0, description="Decimal degrees N")]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0, description="Decimal degrees E")]


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in WGS-84."""

    min_lat: float = Field(..., ge=-90, le=90)
    max_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lon: float = Field(..., ge=-180, le=180)

    @model_validator(mode="after")
    def validate_box(self) -> "BoundingBox":
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        return self


class DataQuality(BaseModel):
    """Data source quality indicators."""

    source: str = Field(..., description="Data origin (e.g. 'BMKG', 'GPM_IMERG')")
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="QA confidence [0-1]")]
    cloud_cover_pct: Optional[float] = Field(None, ge=0, le=100)
    last_updated: datetime


# ---------------------------------------------------------------------------
# 1. Climate current conditions
# ---------------------------------------------------------------------------


class AirQualityIndex(BaseModel):
    aqi: int = Field(..., ge=0, description="Indonesian ISPU AQI value")
    category: str = Field(..., description="Baik / Sedang / Tidak Sehat / Sangat Tidak Sehat / Berbahaya")
    pm25_ugm3: Optional[float] = None
    pm10_ugm3: Optional[float] = None
    o3_ppb: Optional[float] = None
    no2_ppb: Optional[float] = None
    so2_ppb: Optional[float] = None
    co_ppm: Optional[float] = None


class ClimateCurrentResponse(BaseModel):
    """Response for GET /v1/climate/{district_id}/current."""

    district_id: str = Field(..., description="BPS district code (7-digit Kode Wilayah)")
    district_name: str
    province: str
    coordinate: Coordinate
    observed_at: datetime

    # Atmospheric
    temperature_c: float = Field(..., description="2-metre air temperature °C")
    feels_like_c: Optional[float] = None
    relative_humidity_pct: float = Field(..., ge=0, le=100)
    dew_point_c: Optional[float] = None
    pressure_hpa: Optional[float] = None

    # Precipitation
    precipitation_1h_mm: float = Field(default=0.0, ge=0)
    precipitation_24h_mm: float = Field(default=0.0, ge=0)
    precipitation_anomaly_pct: Optional[float] = Field(
        None, description="Departure from 30-year climatological mean (%)"
    )

    # Wind
    wind_speed_ms: Optional[float] = Field(None, ge=0)
    wind_direction_deg: Optional[float] = Field(None, ge=0, le=360)

    # Derived indices
    heat_index_c: Optional[float] = None
    soil_moisture_m3m3: Optional[float] = Field(None, ge=0, le=1, description="SMAP volumetric")
    ndvi: Optional[float] = Field(None, ge=-1, le=1, description="MODIS NDVI latest")
    aqi: Optional[AirQualityIndex] = None

    quality: DataQuality


# ---------------------------------------------------------------------------
# 2. Forecast
# ---------------------------------------------------------------------------


class ForecastHour(BaseModel):
    """Single forecast time-step."""

    valid_time: datetime
    lead_hours: int = Field(..., ge=0)
    temperature_c: float
    feels_like_c: Optional[float] = None
    relative_humidity_pct: float
    precipitation_mm: float = Field(default=0.0, ge=0)
    precipitation_probability_pct: float = Field(default=0.0, ge=0, le=100)
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    cloud_cover_pct: Optional[float] = Field(None, ge=0, le=100)


class ForecastResponse(BaseModel):
    """Response for GET /v1/forecast/{lat}/{lon}."""

    latitude: float
    longitude: float
    district_id: Optional[str] = None
    district_name: Optional[str] = None
    issued_at: datetime
    model: str = Field(default="XGBoost-Precip+NWP", description="Forecast model identifier")
    horizon_hours: int = Field(..., description="Total forecast horizon in hours")
    hourly: list[ForecastHour]
    quality: DataQuality


# ---------------------------------------------------------------------------
# 3. Flood alerts
# ---------------------------------------------------------------------------


class FloodAlert(BaseModel):
    """Single flood alert record."""

    alert_id: str
    alert_type: AlertType = AlertType.FLOOD
    severity: AlertSeverity
    title: str
    description: str

    # Location
    affected_area: str
    affected_districts: list[str] = Field(default_factory=list)
    river_basin: Optional[str] = None
    watershed_id: Optional[str] = None
    latitude: float
    longitude: float
    radius_km: float = Field(..., gt=0)

    # Timing
    issued_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

    # Hydrological context
    peak_streamflow_m3s: Optional[float] = None
    gauge_station: Optional[str] = None
    inundation_area_km2: Optional[float] = None
    affected_population_est: Optional[int] = None

    # Source
    source_agent: str = "HYDROLOGIS"
    source_model: Optional[str] = None


class FloodAlertsResponse(BaseModel):
    """Response for GET /v1/flood/alerts."""

    total: int
    active_count: int
    generated_at: datetime
    alerts: list[FloodAlert]


# ---------------------------------------------------------------------------
# 4. Streamflow
# ---------------------------------------------------------------------------


class StreamflowObservation(BaseModel):
    """Single streamflow reading."""

    timestamp: datetime
    discharge_m3s: float = Field(..., ge=0)
    stage_m: Optional[float] = Field(None, ge=0, description="Water stage / gauge height in metres")
    velocity_ms: Optional[float] = None


class StreamflowThresholds(BaseModel):
    """Alert thresholds for a gauge station."""

    action_m3s: float
    minor_flood_m3s: float
    moderate_flood_m3s: float
    major_flood_m3s: float


class StreamflowStation(BaseModel):
    """Gauge station metadata."""

    station_id: str
    station_name: str
    river_name: str
    basin: str
    latitude: float
    longitude: float
    operator: str = Field(default="BMKG", description="Operating agency")
    elevation_m: Optional[float] = None


class StreamflowResponse(BaseModel):
    """Response for GET /v1/streamflow/{station_id}."""

    station: StreamflowStation
    status: StreamflowStatus
    current_discharge_m3s: float
    current_stage_m: Optional[float] = None
    thresholds: StreamflowThresholds
    observations_24h: list[StreamflowObservation]
    forecast_6h: list[StreamflowObservation] = Field(
        default_factory=list, description="6-hour streamflow forecast from HYDROLOGIS"
    )
    alert: Optional[FloodAlert] = None
    quality: DataQuality


# ---------------------------------------------------------------------------
# 5. Satellite latest
# ---------------------------------------------------------------------------


class SatelliteBand(BaseModel):
    """Single spectral band metadata."""

    name: str = Field(..., description="e.g. 'B4 Red', 'B8 NIR'")
    wavelength_nm: Optional[float] = None
    resolution_m: float
    value: Optional[float] = Field(None, description="Surface reflectance [0-1] or radiance")


class SatelliteScene(BaseModel):
    """Metadata for a single satellite scene/granule."""

    scene_id: str
    mission: SatelliteMission
    acquired_at: datetime
    bbox: BoundingBox
    cloud_cover_pct: float = Field(..., ge=0, le=100)
    resolution_m: float
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    bands: list[SatelliteBand] = Field(default_factory=list)


class DerivedProduct(BaseModel):
    """Pre-computed derived analysis product."""

    product_type: str = Field(
        ...,
        description="e.g. 'NDVI', 'NDWI', 'LST', 'AQI', 'Flood_Extent', 'Burn_Severity'",
    )
    value: Optional[float] = None
    unit: Optional[str] = None
    geotiff_url: Optional[str] = None
    computed_at: datetime
    algorithm: Optional[str] = None


class SatelliteLatestResponse(BaseModel):
    """Response for GET /v1/satellite/latest."""

    latitude: float
    longitude: float
    district_id: Optional[str] = None
    scenes: list[SatelliteScene]
    derived_products: list[DerivedProduct] = Field(default_factory=list)
    generated_at: datetime


# ---------------------------------------------------------------------------
# Error / common response wrappers
# ---------------------------------------------------------------------------


class APIError(BaseModel):
    """Standard error envelope."""

    status_code: int
    error: str
    detail: str
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedMeta(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    total: int
    has_next: bool


# ---------------------------------------------------------------------------
# Auth token response
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token TTL in seconds")
    role: UserRole


class ServiceTokenRequest(BaseModel):
    """Request body for service-to-service token issuance."""

    service_name: str = Field(..., description="Registered service identifier")
    service_secret: str = Field(..., description="Pre-shared service secret")
    role: UserRole = UserRole.PUBLIC
