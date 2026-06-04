"""Climate alerts API routes — flood, fire, deforestation, AQI"""

from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/alerts")


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


class Alert(BaseModel):
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_area: str
    latitude: float
    longitude: float
    radius_km: float
    issued_at: datetime
    expires_at: Optional[datetime] = None
    source_agent: str


class AlertSubscription(BaseModel):
    email: str
    alert_types: list[AlertType]
    provinces: list[str]
    min_severity: AlertSeverity = AlertSeverity.WARNING


@router.get("/active", response_model=list[Alert])
async def get_active_alerts(
    alert_type: Optional[AlertType] = None,
    severity: Optional[AlertSeverity] = None,
    province: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[Alert]:
    """Active climate alerts aggregated from ATMOSPHERE, HYDROLOGIS, GEOSPATIAL."""
    return []


@router.post("/subscribe", status_code=201)
async def subscribe_to_alerts(subscription: AlertSubscription) -> dict:
    """Subscribe to alert notifications via email/webhook."""
    return {"message": "Subscription registered", "id": "sub_placeholder"}


@router.get("/history", response_model=list[Alert])
async def get_alert_history(
    days: int = Query(30, ge=1, le=365),
    alert_type: Optional[AlertType] = None,
) -> list[Alert]:
    """Historical alert records for trend analysis."""
    return []
