"""JWT authentication handler with RBAC support.

Roles:
    admin     — full platform access, all endpoints
    partner   — BMKG, KLHK, LAPAN (read + subscribe)
    public    — read-only public endpoints
    internal  — service-to-service (HYDROLOGIS, ATMOSPHERE, GEOSPATIAL pipelines)
"""

import os
from datetime import datetime, timedelta
from typing import Optional

try:
    import jwt
except ImportError:
    raise ImportError("PyJWT is required: pip install PyJWT")

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JWT_SECRET: str = os.environ.get("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRY_HOURS: int = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))

# FastAPI security scheme — expects: Authorization: Bearer <token>
security = HTTPBearer(auto_error=True)

# Valid role hierarchy (ordered least → most privileged)
VALID_ROLES = {"public", "partner", "internal", "admin"}


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def create_token(
    sub: str,
    role: str,
    expires_in_hours: Optional[int] = None,
    extra_claims: Optional[dict] = None,
) -> str:
    """Sign and return a JWT token for the given subject and role."""
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role '{role}'. Must be one of: {VALID_ROLES}")
    now = datetime.utcnow()
    payload: dict = {
        "sub": sub,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=expires_in_hours or JWT_EXPIRY_HOURS),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------

def decode_token(token: str) -> dict:
    """Decode and validate a JWT; raises HTTP 401 on failure."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """FastAPI dependency — extract and validate JWT from Authorization header."""
    return decode_token(credentials.credentials)


# ---------------------------------------------------------------------------
# RBAC dependency factory
# ---------------------------------------------------------------------------

def require_roles(*allowed_roles: str):
    """
    Returns a FastAPI dependency that enforces role-based access control.

    Usage::

        @router.post("/secure")
        async def endpoint(token = Depends(require_roles("admin", "internal"))):
            ...
    """
    def _role_checker(
        token: dict = Depends(get_current_token),
    ) -> dict:
        role = token.get("role", "")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{role}' is not authorized for this endpoint. "
                    f"Accepted roles: {sorted(allowed_roles)}"
                ),
            )
        return token

    return _role_checker


# ---------------------------------------------------------------------------
# Convenience dependency shortcuts
# ---------------------------------------------------------------------------

#: Only platform admins
require_admin = require_roles("admin")

#: BMKG / KLHK / LAPAN partners and above
require_partner_or_above = require_roles("admin", "partner")

#: Internal pipeline services (HYDROLOGIS, ATMOSPHERE, GEOSPATIAL) and admins
require_internal = require_roles("admin", "internal")

#: Any authenticated caller (all four roles)
require_authenticated = require_roles("admin", "partner", "internal", "public")
