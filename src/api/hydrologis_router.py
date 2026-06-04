"""
HYDROLOGIS API Router — Sprint 3 Deliverable 5
FastAPI router for HYDROLOGIS hydrological outputs.

Prefix: /api/v1/hydrologis
Base SHA: ANALYTICA c418a904 (model_server.py)

Endpoints:
  GET /streamflow/{river_id}/forecast   — latest StreamflowForecast JSON
  GET /flood-extent/{river_id}          — latest flood extent GeoJSON
  GET /agri-advisory/{watershed_id}     — latest AgriWaterAdvisory JSON
  GET /aquifer/{aquifer_id}/latest      — latest AquiferReport JSON

All endpoints:
  - Cache-Control: max-age=1800 (30 minutes)
  - 404 with structured error if output file not found
  - Prometheus counter: tropi_hydrologis_api_requests_total{endpoint, status_code}

Wire into model_server.py:
  from src.api.hydrologis_router import router as hydrologis_router
  app.include_router(hydrologis_router)
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counter
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter
    from src.hydrology.metrics import _REGISTRY

    API_REQUESTS_COUNTER = Counter(
        "tropi_hydrologis_api_requests_total",
        "Total HTTP requests served by the HYDROLOGIS API router",
        labelnames=["endpoint", "status_code"],
        registry=_REGISTRY,
    )
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False
    class _StubCounter:
        def labels(self, **_): return self
        def inc(self): pass
    API_REQUESTS_COUNTER = _StubCounter()   # type: ignore[assignment]


def _count(endpoint: str, status_code: int) -> None:
    API_REQUESTS_COUNTER.labels(endpoint=endpoint, status_code=str(status_code)).inc()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE   = os.getenv("WORKSPACE_ROOT", "workspace")
OUTPUT_DIRS = {
    "streamflow":  os.path.join(WORKSPACE, "output", "streamflow"),
    "flood_extent": os.path.join(WORKSPACE, "output", "flood_extent"),
    "agri":        os.path.join(WORKSPACE, "output", "agri_advisory"),
    "aquifer":     os.path.join(WORKSPACE, "output", "aquifer"),
}
CACHE_MAX_AGE = 1800   # seconds (30 minutes)
CACHE_HEADER  = f"public, max-age={CACHE_MAX_AGE}"

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/hydrologis",
    tags=["hydrologis"],
)


# ---------------------------------------------------------------------------
# Helper: find latest output file
# ---------------------------------------------------------------------------

def _latest_file(directory: str, pattern: str) -> Optional[str]:
    """Return path to the most-recently modified file matching pattern, or None."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _not_found(resource: str, id_: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "error":    "NOT_FOUND",
            "message":  f"No {resource} output found for '{id_}'.",
            "resource": resource,
            "id":       id_,
            "hint":     "Run the relevant pipeline first or wait for the next scheduled cycle.",
        },
    )


# ---------------------------------------------------------------------------
# Endpoint 1 — Streamflow forecast
# ---------------------------------------------------------------------------

@router.get(
    "/streamflow/{river_id}/forecast",
    summary="Latest streamflow forecast for a river",
    response_description="StreamflowForecast JSON (all horizons)",
)
async def get_streamflow_forecast(river_id: str) -> Response:
    """
    Return the latest ensemble streamflow forecast (6/12/24h) for the given river.

    Valid river_id values: ciliwung, brantas, solo
    """
    endpoint = "streamflow_forecast"
    path = _latest_file(OUTPUT_DIRS["streamflow"], f"{river_id}_*.json")
    if path is None:
        _count(endpoint, 404)
        raise _not_found("streamflow_forecast", river_id)

    try:
        data = _load_json(path)
        _count(endpoint, 200)
        return Response(
            content=json.dumps(data),
            media_type="application/json",
            headers={"Cache-Control": CACHE_HEADER},
        )
    except Exception as exc:
        logger.exception("Error loading streamflow forecast for %s: %s", river_id, exc)
        _count(endpoint, 500)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoint 2 — Flood extent
# ---------------------------------------------------------------------------

@router.get(
    "/flood-extent/{river_id}",
    summary="Latest flood inundation GeoJSON for a river",
    response_description="GeoJSON FeatureCollection with flood polygon(s)",
)
async def get_flood_extent(
    river_id: str,
    horizon_hours: Optional[int] = None,
) -> Response:
    """
    Return the latest flood inundation GeoJSON for the given river.

    Optional query param:
      horizon_hours (int) — filter to a specific forecast horizon (6, 12, or 24).
      If omitted, returns the 24h horizon by default.
    """
    endpoint = "flood_extent"
    h = horizon_hours or 24
    path = _latest_file(OUTPUT_DIRS["flood_extent"], f"{river_id}_{h}h_*.geojson")

    if path is None:
        _count(endpoint, 404)
        raise _not_found("flood_extent", f"{river_id} (horizon={h}h)")

    try:
        data = _load_json(path)
        _count(endpoint, 200)
        return Response(
            content=json.dumps(data),
            media_type="application/geo+json",
            headers={"Cache-Control": CACHE_HEADER},
        )
    except Exception as exc:
        logger.exception("Error loading flood extent for %s: %s", river_id, exc)
        _count(endpoint, 500)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoint 3 — Agricultural water advisory
# ---------------------------------------------------------------------------

@router.get(
    "/agri-advisory/{watershed_id}",
    summary="Latest agricultural water availability advisory for a watershed",
    response_description="AgriWaterAdvisory JSON (30/60/90-day horizons)",
)
async def get_agri_advisory(
    watershed_id: str,
    horizon_days: Optional[int] = None,
) -> Response:
    """
    Return the latest seasonal irrigation advisory for the given watershed.

    Valid watershed_id examples: das_ciliwung, das_brantas, das_bengawan_solo, ...
    Optional query param:
      horizon_days (int) — filter to 30, 60, or 90-day horizon.
      If omitted, returns all horizons.
    """
    endpoint = "agri_advisory"
    # Find latest monthly advisory file
    path = _latest_file(OUTPUT_DIRS["agri"], "agri_water_advisory_*.json")
    if path is None:
        _count(endpoint, 404)
        raise _not_found("agri_water_advisory", watershed_id)

    try:
        data = _load_json(path)
        # Filter to requested watershed
        advisories = [
            a for a in data.get("advisories", [])
            if a.get("watershed_id") == watershed_id
        ]
        if not advisories:
            _count(endpoint, 404)
            raise _not_found("agri_water_advisory", watershed_id)
        if horizon_days:
            advisories = [a for a in advisories if a.get("advisory_horizon_days") == horizon_days]
        result = {**data, "advisories": advisories}
        _count(endpoint, 200)
        return Response(
            content=json.dumps(result, default=str),
            media_type="application/json",
            headers={"Cache-Control": CACHE_HEADER},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error loading agri advisory for %s: %s", watershed_id, exc)
        _count(endpoint, 500)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(exc)})


# ---------------------------------------------------------------------------
# Endpoint 4 — Aquifer depletion report
# ---------------------------------------------------------------------------

@router.get(
    "/aquifer/{aquifer_id}/latest",
    summary="Latest GRACE-FO groundwater storage anomaly report for an aquifer",
    response_description="AquiferReport JSON",
)
async def get_aquifer_latest(aquifer_id: str) -> Response:
    """
    Return the latest GRACE-FO derived groundwater depletion report.

    Valid aquifer_id values: north_jakarta, bandung_basin, semarang, surabaya, makassar
    """
    endpoint = "aquifer_latest"
    path = _latest_file(OUTPUT_DIRS["aquifer"], f"{aquifer_id}_*.json")
    if path is None:
        _count(endpoint, 404)
        raise _not_found("aquifer_report", aquifer_id)

    try:
        data = _load_json(path)
        _count(endpoint, 200)
        return Response(
            content=json.dumps(data, default=str),
            media_type="application/json",
            headers={"Cache-Control": CACHE_HEADER},
        )
    except Exception as exc:
        logger.exception("Error loading aquifer report for %s: %s", aquifer_id, exc)
        _count(endpoint, 500)
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(exc)})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="HYDROLOGIS router health check",
    include_in_schema=False,
)
async def health() -> JSONResponse:
    return JSONResponse(
        content={
            "status": "ok",
            "agent": "HYDROLOGIS",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        headers={"Cache-Control": "no-store"},
    )
