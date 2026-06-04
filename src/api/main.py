"""FastAPI application — Tropi Climate Analytics API v1.

Endpoints
---------
GET  /v1/climate/{district_id}/current   Current climate conditions for an Indonesian district
GET  /v1/forecast/{lat}/{lon}            72-hour meteorological forecast
GET  /v1/flood/alerts                    Active flood early-warning alerts (HYDROLOGIS)
GET  /v1/streamflow/{station_id}         Real-time streamflow + 6-hour forecast
GET  /v1/satellite/latest                Latest satellite scenes + derived products

Authentication
--------------
JWT Bearer (HS256). Role hierarchy:
    public < researcher < government < admin

Rate limits (enforced at Kong Gateway layer — see kong_config.yaml):
    public:      100  req/hr
    researcher:  1000 req/hr
    government: 10000 req/hr
    admin:       unlimited
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Annotated, Any, Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.schemas import (
    AlertSeverity,
    AlertType,
    ClimateCurrentResponse,
    FloodAlert,
    FloodAlertsResponse,
    ForecastResponse,
    SatelliteLatestResponse,
    ServiceTokenRequest,
    StreamflowResponse,
    TokenResponse,
    UserRole,
)
from src.api.openapi_extensions import (
    OPENAPI_DESCRIPTION,
    OPENAPI_TAGS,
    PARTNER_CONTACT,
    build_custom_openapi,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JWT_SECRET: str = os.environ.get("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRY_HOURS: int = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))

# Role ordering (higher index = more privileged)
ROLE_ORDER: dict[str, int] = {
    UserRole.PUBLIC: 0,
    UserRole.RESEARCHER: 1,
    UserRole.GOVERNMENT: 2,
    UserRole.ADMIN: 3,
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tropi Climate Analytics API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — allow partner portals (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS",
        "https://portal.bmkg.go.id,https://klhk.go.id,https://lapan.go.id",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# HYDROLOGIS hydrological model router
# NOTE: _REGISTRY is instantiated inside hydrologis_router.py — not re-instantiated here.
from src.api.hydrologis_router import router as hydrologis_router
app.include_router(hydrologis_router)

# ANALYTICA inference router (Sprint 6 H1)
# Endpoints: POST /predict/{precipitation,seasonal,landcover,streamflow,climate}
# Health:    GET  /predict/healthz/inference
from src.serving.inference_api import router as inference_router
app.include_router(inference_router)


# Custom OpenAPI schema with partner metadata
@app.on_event("startup")
async def _patch_openapi() -> None:
    app.openapi_schema = build_custom_openapi(app)


# ---------------------------------------------------------------------------
# JWT auth & RBAC middleware
# ---------------------------------------------------------------------------

_http_bearer = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _get_optional_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_http_bearer),
) -> dict | None:
    """Return decoded token if present, else None (for public endpoints)."""
    if credentials is None:
        return None
    return _decode_token(credentials.credentials)


def _get_required_token(
    credentials: HTTPAuthorizationCredentials = Security(_http_bearer),
) -> dict:
    """Require a valid JWT; raise 401 if missing or invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(credentials.credentials)


def require_role(minimum_role: UserRole):
    """
    FastAPI dependency factory for RBAC.

    Usage::

        @router.get("/sensitive")
        async def endpoint(token = Depends(require_role(UserRole.GOVERNMENT))):
            ...
    """
    def _checker(
        token: dict = Depends(_get_required_token),
    ) -> dict:
        role_str = token.get("role", UserRole.PUBLIC)
        caller_level = ROLE_ORDER.get(role_str, 0)
        required_level = ROLE_ORDER[minimum_role]
        if caller_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{role_str}' insufficient. "
                    f"Minimum required: '{minimum_role.value}'"
                ),
            )
        return token

    return _checker


# Convenience aliases
require_public = require_role(UserRole.PUBLIC)
require_researcher = require_role(UserRole.RESEARCHER)
require_government = require_role(UserRole.GOVERNMENT)
require_admin = require_role(UserRole.ADMIN)


# ---------------------------------------------------------------------------
# Request-ID middleware (echoes Kong-injected X-Request-ID)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")
    response = await call_next(request)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Health / meta endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Meta"],
    summary="Health check",
    description="Returns 200 when the API is healthy. Used by Kong and load-balancer probes.",
)
async def health_check() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get(
    "/v1/info",
    tags=["Meta"],
    summary="API version and capability info",
)
async def api_info() -> dict:
    return {
        "version": "1.0.0",
        "description": "Tropi Climate Analytics API",
        "endpoints": [
            "GET /v1/climate/{district_id}/current",
            "GET /v1/forecast/{lat}/{lon}",
            "GET /v1/flood/alerts",
            "GET /v1/streamflow/{station_id}",
            "GET /v1/satellite/latest",
        ],
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }


# ---------------------------------------------------------------------------
# Auth endpoint (token issuance — for service-to-service and dev usage)
# ---------------------------------------------------------------------------

@app.post(
    "/v1/auth/token",
    response_model=TokenResponse,
    tags=["Auth"],
    summary="Issue a JWT",
    description=(
        "Issues a signed JWT for the requesting service or user. "
        "In production this is backed by the identity provider — "
        "this stub is for integration testing only."
    ),
)
async def issue_token(body: ServiceTokenRequest) -> TokenResponse:
    # TODO: validate service_name + service_secret against credential store
    now = datetime.utcnow()
    payload = {
        "sub": body.service_name,
        "role": body.role.value,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=JWT_EXPIRY_HOURS * 3600,
        role=body.role,
    )


# ---------------------------------------------------------------------------
# 1. Climate current conditions
# ---------------------------------------------------------------------------

@app.get(
    "/v1/climate/{district_id}/current",
    response_model=ClimateCurrentResponse,
    tags=["Climate"],
    summary="Current climate conditions for an Indonesian district",
    description=(
        "Returns the latest observed climate variables for the requested district "
        "(temperature, humidity, precipitation, soil moisture, NDVI, AQI). "
        "Data fused from BMKG ground observations + MODIS/SMAP satellite retrievals. "
        "Redis TTL: 300 s.  |  **Rate limit:** public 100/hr, researcher 1000/hr, "
        "government 10000/hr."
    ),
)
async def get_climate_current(
    district_id: str,
    include_aqi: bool = Query(True, description="Include ISPU AQI data"),
    include_satellite: bool = Query(True, description="Include MODIS NDVI / SMAP soil moisture"),
    token: dict | None = Depends(_get_optional_token),
) -> ClimateCurrentResponse:
    """
    Public endpoint — no auth required, but authenticated callers get
    higher rate limits and additional fields (researcher: raw QA scores;
    government: full sensor metadata).

    TODO: query Redis cache → if miss, query TimescaleDB + PostGIS pipeline.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Connect data pipeline")


# ---------------------------------------------------------------------------
# 2. Forecast
# ---------------------------------------------------------------------------

@app.get(
    "/v1/forecast/{lat}/{lon}",
    response_model=ForecastResponse,
    tags=["Forecast"],
    summary="72-hour meteorological forecast",
    description=(
        "XGBoost-based 24–72 hour precipitation and temperature forecast for the "
        "requested WGS-84 coordinate. Produced by ANALYTICA; served with a "
        "15-minute Redis cache.  |  **Min role:** public."
    ),
)
async def get_forecast(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    hours: int = Query(72, ge=6, le=240, description="Forecast horizon in hours"),
    token: dict | None = Depends(_get_optional_token),
) -> ForecastResponse:
    """
    TODO: forward to ANALYTICA forecast service / read from feature store.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Connect forecast model")


# ---------------------------------------------------------------------------
# 3. Flood alerts
# ---------------------------------------------------------------------------

@app.get(
    "/v1/flood/alerts",
    response_model=FloodAlertsResponse,
    tags=["Flood & Hydrology"],
    summary="Active flood early-warning alerts",
    description=(
        "Returns all currently active flood alerts ingested from HYDROLOGIS "
        "FloodEarlyWarningPipeline via POST /v1/alerts/active. "
        "Filterable by severity, province, and river basin. "
        "Redis TTL: 60 s.  |  **Min role:** public."
    ),
)
async def get_flood_alerts(
    severity: Optional[AlertSeverity] = Query(None),
    province: Optional[str] = Query(None, description="Indonesian province name"),
    river_basin: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    token: dict | None = Depends(_get_optional_token),
) -> FloodAlertsResponse:
    """
    TODO: query active alerts from TimescaleDB alerts table with filters.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Connect alerts store")


# ---------------------------------------------------------------------------
# 4. Streamflow
# ---------------------------------------------------------------------------

@app.get(
    "/v1/streamflow/{station_id}",
    response_model=StreamflowResponse,
    tags=["Flood & Hydrology"],
    summary="Real-time streamflow + 6-hour forecast for a gauge station",
    description=(
        "Returns real-time discharge, water stage, 24-hour observation history, "
        "and HYDROLOGIS 6-hour streamflow forecast for the requested BMKG gauge station. "
        "Includes flood threshold values and active alert if threshold exceeded. "
        "Redis TTL: 30 s.  |  **Min role:** public."
    ),
)
async def get_streamflow(
    station_id: str,
    include_forecast: bool = Query(True, description="Include HYDROLOGIS 6-hour forecast"),
    token: dict | None = Depends(_get_optional_token),
) -> StreamflowResponse:
    """
    TODO: query HYDROLOGIS streamflow API or TimescaleDB hydrological time series.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Connect streamflow store")


# ---------------------------------------------------------------------------
# 5. Satellite latest
# ---------------------------------------------------------------------------

@app.get(
    "/v1/satellite/latest",
    response_model=SatelliteLatestResponse,
    tags=["Satellite"],
    summary="Latest satellite scenes + derived products",
    description=(
        "Returns the most recent cloud-free satellite scenes for the requested coordinate "
        "across configured missions (Landsat 8/9, MODIS Terra/Aqua, SMAP, GPM-IMERG, Sentinel-2). "
        "Also returns pre-computed derived products: NDVI, NDWI, LST, flood extent, burn severity. "
        "Researcher+ required for raw scene download URLs.  |  **Min role:** public (metadata), "
        "researcher (download URLs)."
    ),
)
async def get_satellite_latest(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    missions: Optional[list[str]] = Query(
        None, description="Filter to specific satellite missions"
    ),
    max_cloud_cover: float = Query(30.0, ge=0, le=100, description="Max cloud cover %"),
    days_back: int = Query(14, ge=1, le=90, description="How far back to search for scenes"),
    token: dict | None = Depends(_get_optional_token),
) -> SatelliteLatestResponse:
    """
    - Public: scene metadata + derived product values only
    - Researcher+: scene preview_url + download_url included
    - TODO: query DATA-FLOW scene catalogue (NASA CMR + internal S3 index)
    """
    # Enforce download URL gate
    role = (token or {}).get("role", UserRole.PUBLIC)
    include_urls = ROLE_ORDER.get(role, 0) >= ROLE_ORDER[UserRole.RESEARCHER]

    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Connect satellite catalogue")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "error": exc.detail,
            "detail": str(exc.detail),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
