"""Climate alerts API routes – flood, fire, deforestation, AQI"""

from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.auth.jwt_handler import require_internal, require_authenticated

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_area: str = Field(..., description="Human-readable area name (province / district)")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(..., gt=0, description="Affected radius in kilometres")
    issued_at: datetime
    expires_at: Optional[datetime] = None
    source_agent: str = Field(..., description="Originating pipeline agent (e.g. HYDROLOGIS)")
    is_active: bool = True


class AlertIngestion(BaseModel):
    """
    Payload accepted from internal pipeline services.

    Callers: HYDROLOGIS FloodEarlyWarningPipeline, ATMOSPHERE AQI pipeline,
             GEOSPATIAL deforestation detector.
    Auth   : Bearer JWT with role='internal' or role='admin'.
    """
    alert_id: str = Field(..., description="Caller-generated UUID for idempotent ingestion")
    alert_type: AlertType
    severity: AlertSeverity
    title: str = Field(..., max_length=200)
    description: str = Field(..., max_length=2000)
    affected_area: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(..., gt=0)
    issued_at: datetime
    expires_at: Optional[datetime] = None
    source_agent: str = Field(
        ...,
        description="Pipeline identifier posting this alert (e.g. 'HYDROLOGIS')",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional structured payload (streamflow values, AQI readings, etc.)",
    )


class AlertResolveRequest(BaseModel):
    """Optional body for resolve endpoint."""
    reason: Optional[str] = Field(None, description="Human-readable resolution reason")


class AlertResolveResponse(BaseModel):
    alert_id: str
    resolved_at: datetime
    resolved_by: str = Field(..., description="JWT subject (service name) that resolved the alert")
    message: str


class AlertSubscription(BaseModel):
    email: str
    alert_types: list[AlertType]
    provinces: list[str]
    min_severity: AlertSeverity = AlertSeverity.WARNING


# ---------------------------------------------------------------------------
# Public read endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/active",
    response_model=list[Alert],
    summary="List active climate alerts",
    description="Returns all currently active alerts. Filterable by type, severity, and province.",
)
async def get_active_alerts(
    alert_type: Optional[AlertType] = Query(None, description="Filter by alert category"),
    severity: Optional[AlertSeverity] = Query(None, description="Minimum severity filter"),
    province: Optional[str] = Query(None, description="Indonesian province name"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[Alert]:
    """Active climate alerts aggregated from ATMOSPHERE, HYDROLOGIS, GEOSPATIAL."""
    # TODO: query database with filters
    return []


@router.get(
    "/history",
    response_model=list[Alert],
    summary="Alert history",
    description="Historical alert records for trend analysis.",
)
async def get_alert_history(
    days: int = Query(30, ge=1, le=365),
    alert_type: Optional[AlertType] = None,
) -> list[Alert]:
    """Historical alert records for trend analysis."""
    # TODO: query database
    return []


# ---------------------------------------------------------------------------
# Internal service-to-service ingestion endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/active",
    response_model=Alert,
    status_code=status.HTTP_201_CREATED,
    summary="[INTERNAL] Ingest a new alert from a pipeline service",
    description=(
        "**Service-to-service endpoint.** Accepts alert payloads from internal pipeline "
        "agents (HYDROLOGIS FloodEarlyWarningPipeline, ATMOSPHERE, GEOSPATIAL). "
        "Persists the alert and triggers the webhook dispatcher for all active subscribers. "
        "Requires a JWT with `role='internal'` or `role='admin'`.  "
        "Partners and public callers receive **HTTP 403**."
    ),
)
async def ingest_alert(
    payload: AlertIngestion,
    token: dict = Depends(require_internal),
) -> Alert:
    """
    Ingest a new alert from an internal service.

    Authentication
    --------------
    Bearer JWT issued to the calling service with::

        { "sub": "hydrologis-pipeline", "role": "internal" }

    Kong rate limit: 5000 req/min (internal tier).

    Webhook dispatch
    ----------------
    On success, the alert is forwarded to all matching webhook subscriptions
    via ``src.api.webhooks.dispatcher.dispatch_alert``.
    """
    # TODO: persist to TimescaleDB / PostGIS alerts table
    # TODO: await dispatcher.dispatch_alert(alert)
    alert = Alert(
        alert_id=payload.alert_id,
        alert_type=payload.alert_type,
        severity=payload.severity,
        title=payload.title,
        description=payload.description,
        affected_area=payload.affected_area,
        latitude=payload.latitude,
        longitude=payload.longitude,
        radius_km=payload.radius_km,
        issued_at=payload.issued_at,
        expires_at=payload.expires_at,
        source_agent=payload.source_agent,
        is_active=True,
    )
    return alert


@router.post(
    "/{alert_id}/resolve",
    response_model=AlertResolveResponse,
    summary="[INTERNAL] Mark an alert as resolved",
    description=(
        "**Service-to-service endpoint.** Marks an active alert as resolved/inactive. "
        "Triggers a resolution notification to all subscribers who received the original alert. "
        "Requires a JWT with `role='internal'` or `role='admin'`."
    ),
)
async def resolve_alert(
    alert_id: str,
    body: Optional[AlertResolveRequest] = None,
    token: dict = Depends(require_internal),
) -> AlertResolveResponse:
    """
    Mark an active alert as resolved.

    Called by:
        - HYDROLOGIS when streamflow drops below flood threshold
        - ATMOSPHERE when AQI returns to safe level
        - Admin manual resolution

    Authentication
    --------------
    Bearer JWT with role='internal' or role='admin'.
    """
    # TODO: update is_active=False in database
    # TODO: dispatch resolution webhook notification
    caller = token.get("sub", "unknown-service")
    resolved_at = datetime.utcnow()
    reason = body.reason if body and body.reason else "Resolved by pipeline"
    return AlertResolveResponse(
        alert_id=alert_id,
        resolved_at=resolved_at,
        resolved_by=caller,
        message=f"Alert '{alert_id}' resolved by '{caller}'. Reason: {reason}",
    )


# ---------------------------------------------------------------------------
# Subscription management
# ---------------------------------------------------------------------------

@router.post(
    "/subscribe",
    status_code=status.HTTP_201_CREATED,
    summary="Subscribe to alert notifications",
)
async def subscribe_to_alerts(subscription: AlertSubscription) -> dict:
    """Subscribe to alert notifications via email/webhook."""
    # TODO: persist subscription + trigger confirm email
    return {"message": "Subscription registered", "id": "sub_placeholder"}


@router.delete(
    "/subscribe/{subscription_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel alert subscription",
)
async def unsubscribe_from_alerts(subscription_id: str) -> dict:
    """Cancel an existing alert subscription."""
    # TODO: soft-delete subscription record
    return {"message": f"Subscription {subscription_id} cancelled"}
