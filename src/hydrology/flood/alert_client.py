"""
Flood Alert API Client — Authenticated HTTP client for HYDROLOGIS pipeline
Agent: HYDROLOGIS | Tropi Climate Analytics

Integrates with Sprint 1 endpoints delivered by API-GATEWAY:
  POST /api/v1/alerts/active          — ingest flood alert (HTTP 201)
  POST /api/v1/alerts/{id}/resolve    — resolve alert (HTTP 200)

Auth flow:
  POST /v1/auth/token  {sub: hydrologis-pipeline, role: internal}
  → JWT HS256 token cached with TTL < exp (refreshed 60s before expiry)

Config (env vars):
  TROPI_API_BASE_URL   — base URL of the Tropi Climate API (default: http://localhost:8000)
  JWT_SECRET           — shared HS256 secret (never hardcode)
  JWT_EXPIRY_HOURS     — token lifetime in hours (default: 24)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("TROPI_API_BASE_URL", "http://localhost:8000")
_TOKEN_REFRESH_BUFFER_S: int = 60   # Refresh token this many seconds before expiry
_DEFAULT_TIMEOUT_S: float = 15.0


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AlertIngestResponse(BaseModel):
    alert_id: str
    status: str = "created"
    created_at: Optional[datetime] = None
    source_agent: Optional[str] = None


class AlertResolveResponse(BaseModel):
    alert_id: str
    resolved: bool
    resolved_at: Optional[datetime] = None
    reason: Optional[str] = None


class AlertPushResult(BaseModel):
    alert_id: str
    success: bool
    status_code: int
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Token cache (module-level singleton — one token per process)
# ---------------------------------------------------------------------------

class _TokenCache:
    """Thread-safe JWT token cache with automatic refresh before expiry."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._expires_at: float = 0.0   # Unix timestamp

    def is_valid(self) -> bool:
        return (
            self._token is not None
            and time.monotonic() < self._expires_at - _TOKEN_REFRESH_BUFFER_S
        )

    def set(self, token: str, exp_unix: float) -> None:
        self._token = token
        self._expires_at = exp_unix - time.time() + time.monotonic()  # monotonic-aligned

    @property
    def token(self) -> Optional[str]:
        return self._token


_token_cache = _TokenCache()


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

class AlertAPIClient:
    """
    Authenticated HTTP client for HYDROLOGIS → Tropi Climate API alert ingestion.

    Usage:
        client = AlertAPIClient()
        result = client.push_alert(payload_dict)
        client.resolve_alert("flood-ciliwung-20260604-001", "River dropped below threshold")
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self._subject = "hydrologis-pipeline"
        self._role = "internal"

    # ------------------------------------------------------------------
    # JWT token management
    # ------------------------------------------------------------------

    def _obtain_token(self) -> str:
        """
        Obtain a service-to-service JWT token from the auth endpoint.
        Cached with TTL; refreshed 60s before expiry.
        """
        if _token_cache.is_valid():
            return _token_cache.token  # type: ignore[return-value]

        logger.debug("Requesting new JWT token for HYDROLOGIS pipeline")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/auth/token",
                    json={"sub": self._subject, "role": self._role},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                token: str = data["access_token"]

                # Decode exp claim (no verification — we just issued it)
                import base64, json as _json
                parts = token.split(".")
                padding = "=" * (-len(parts[1]) % 4)
                claims = _json.loads(base64.urlsafe_b64decode(parts[1] + padding))
                exp_unix = float(claims.get("exp", time.time() + 86400))

                _token_cache.set(token, exp_unix)
                logger.info("JWT token obtained and cached for hydrologis-pipeline")
                return token

        except httpx.HTTPStatusError as exc:
            logger.error(f"Token request failed HTTP {exc.response.status_code}: {exc.response.text}")
            raise
        except Exception as exc:
            logger.error(f"Token request exception: {exc}")
            raise

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._obtain_token()}",
            "Content-Type": "application/json",
            "X-Source-Agent": "HYDROLOGIS",
        }

    # ------------------------------------------------------------------
    # Alert push
    # ------------------------------------------------------------------

    def push_alert(self, payload: dict) -> AlertPushResult:
        """
        POST payload to /api/v1/alerts/active.

        payload must match AlertIngestion schema:
          alert_id, alert_type, severity, title, description,
          affected_area, latitude, longitude, radius_km,
          issued_at, source_agent, metadata (optional)
        """
        alert_id = payload.get("alert_id", "unknown")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/v1/alerts/active",
                    json=payload,
                    headers=self._auth_headers(),
                )
                success = resp.status_code == 201
                if not success:
                    logger.warning(
                        f"Alert push non-201: {alert_id} → HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    logger.info(f"Alert ingested: {alert_id} → HTTP 201")
                return AlertPushResult(
                    alert_id=alert_id,
                    success=success,
                    status_code=resp.status_code,
                    message=resp.text[:200] if not success else None,
                )
        except httpx.HTTPStatusError as exc:
            logger.error(f"Alert push HTTP error {exc.response.status_code} for {alert_id}")
            return AlertPushResult(
                alert_id=alert_id, success=False,
                status_code=exc.response.status_code,
                message=str(exc)[:200],
            )
        except Exception as exc:
            logger.error(f"Alert push exception for {alert_id}: {exc}")
            return AlertPushResult(
                alert_id=alert_id, success=False,
                status_code=0, message=str(exc)[:200],
            )

    def push_alerts(self, payloads: list[dict]) -> list[AlertPushResult]:
        """Push multiple alerts; token is obtained once and reused."""
        results = []
        for payload in payloads:
            results.append(self.push_alert(payload))
        n_ok = sum(1 for r in results if r.success)
        logger.info(f"Alert batch: {n_ok}/{len(results)} pushed successfully")
        return results

    # ------------------------------------------------------------------
    # Alert resolve
    # ------------------------------------------------------------------

    def resolve_alert(
        self,
        alert_id: str,
        reason: str = "Conditions returned to normal",
    ) -> AlertResolveResponse:
        """
        POST /api/v1/alerts/{alert_id}/resolve to mark alert inactive.
        Called by FloodEarlyWarningPipeline when river stage drops below threshold.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/v1/alerts/{alert_id}/resolve",
                    json={"reason": reason},
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                logger.info(f"Alert resolved: {alert_id} → HTTP {resp.status_code}")
                return AlertResolveResponse(
                    alert_id=alert_id,
                    resolved=True,
                    resolved_at=datetime.now(timezone.utc),
                    reason=reason,
                )
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Resolve HTTP {exc.response.status_code} for {alert_id}: {exc.response.text[:200]}"
            )
            return AlertResolveResponse(alert_id=alert_id, resolved=False, reason=str(exc))
        except Exception as exc:
            logger.error(f"Resolve exception for {alert_id}: {exc}")
            return AlertResolveResponse(alert_id=alert_id, resolved=False, reason=str(exc))
