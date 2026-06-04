"""Custom OpenAPI metadata for Tropi Climate Analytics partner documentation.

Imported by main.py to inject partner-specific descriptions, tags, contact
info, and external documentation links into the generated OpenAPI schema.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

# ---------------------------------------------------------------------------
# Partner / external documentation links
# ---------------------------------------------------------------------------

PARTNER_CONTACT = {
    "name": "API-GATEWAY Support — Tropi Climate Analytics",
    "url": "https://github.com/KOSASIH/tropi-climate-analytics",
    "email": "api-support@tropi-climate.id",
}

PARTNER_LICENSE = {
    "name": "Government Data License — Indonesia (PP 71/2019)",
    "url": "https://peraturan.go.id/peraturan/view.html?id=11e9aa572e8898e0a5ba313134333439",
}

EXTERNAL_DOCS = {
    "description": "Full developer guide & partner integration handbook",
    "url": "https://docs.tropi-climate.id",
}

# ---------------------------------------------------------------------------
# OpenAPI description (Markdown, shown at top of Swagger/ReDoc)
# ---------------------------------------------------------------------------

OPENAPI_DESCRIPTION = """\
# Tropi Climate Analytics API

Programmatic access to near-real-time tropical climate data for Indonesia, \
integrating **NASA satellite** observations (Landsat 8/9, MODIS, SMAP, GPM-IMERG) \
with **BMKG**, **KLHK**, and **LAPAN** ground-station networks.

## Quickstart

```bash
# 1 — Obtain a token (development only; use IdP in production)
TOKEN=$(curl -s -X POST https://api.tropi-climate.id/v1/auth/token \\
  -H "Content-Type: application/json" \\
  -d '{"service_name":"my-app","service_secret":"***","role":"researcher"}' \\
  | jq -r .access_token)

# 2 — Query current climate for Jakarta Selatan (district 3174)
curl -H "Authorization: Bearer $TOKEN" \\
  "https://api.tropi-climate.id/v1/climate/3174/current"

# 3 — Get flood alerts in West Java
curl "https://api.tropi-climate.id/v1/flood/alerts?province=Jawa+Barat"
```

## Authentication

All endpoints except `/health` and public GET routes accept an optional \
JWT Bearer token. Tokens carry a `role` claim:

| Role | Issued to | Rate limit |
|---|---|---|
| `public` | Anonymous / dev | 100 req/hr |
| `researcher` | Academic / NGO | 1 000 req/hr |
| `government` | BMKG, KLHK, LAPAN, Dinas | 10 000 req/hr |
| `admin` | Platform operators | Unlimited |

Tokens are issued at `POST /v1/auth/token` (integration testing) or via \
the production OIDC provider at `https://id.tropi-climate.id`.

## Data Sources

| Variable | Source | Update Frequency |
|---|---|---|
| Temperature / Humidity | BMKG AWOS network | 10 min |
| Precipitation | GPM-IMERG + BMKG ARG | 30 min |
| Soil Moisture | SMAP L3 | 3 days |
| Vegetation (NDVI) | MODIS MOD13A3 | 16 days |
| Flood alerts | HYDROLOGIS FloodEarlyWarningPipeline | Real-time |
| Streamflow | BMKG hydrological gauges | 15 min |
| Satellite scenes | NASA CMR / Landsat, Sentinel-2 | Per overpass |

## Partner Integration

BMKG, KLHK, and LAPAN partners receive `government`-role tokens with \
10 000 req/hr rate limits. To onboard a new partner organisation, \
contact the API-GATEWAY team.

## Compliance

Data handling complies with **PP Nomor 71 Tahun 2019** (Government Regulation \
on Implementation of Electronic Systems and Transactions) and NASA's \
[Open Data policy](https://www.nasa.gov/open/). All API interactions are \
logged with immutable audit trails.
"""

# ---------------------------------------------------------------------------
# OpenAPI tags with descriptions
# ---------------------------------------------------------------------------

OPENAPI_TAGS: list[dict] = [
    {
        "name": "Climate",
        "description": (
            "Current climate conditions fused from BMKG ground observations and "
            "NASA satellite retrievals (MODIS, SMAP). Covers all 514 Indonesian districts."
        ),
        "externalDocs": {
            "description": "Climate data methodology",
            "url": "https://docs.tropi-climate.id/data/climate",
        },
    },
    {
        "name": "Forecast",
        "description": (
            "24–72 hour meteorological forecasts produced by ANALYTICA's XGBoost + "
            "NWP ensemble model. 4 km spatial resolution, updated every 3 hours."
        ),
        "externalDocs": {
            "description": "Forecast model documentation",
            "url": "https://docs.tropi-climate.id/models/forecast",
        },
    },
    {
        "name": "Flood & Hydrology",
        "description": (
            "Flood early-warning alerts and real-time streamflow data from HYDROLOGIS. "
            "Covers major Indonesian river basins: Ciliwung, Brantas, Solo, Citarum, Kapuas. "
            "6–24 hour flood early-warning lead time."
        ),
        "externalDocs": {
            "description": "Flood alert schema & severity levels",
            "url": "https://docs.tropi-climate.id/alerts/flood",
        },
    },
    {
        "name": "Satellite",
        "description": (
            "Latest cloud-free satellite scenes and pre-computed derived products (NDVI, NDWI, "
            "LST, flood extent, burn severity) from Landsat 8/9, MODIS, SMAP, GPM-IMERG, "
            "and Sentinel-2. Sourced via NASA CMR API."
        ),
        "externalDocs": {
            "description": "Satellite product catalogue",
            "url": "https://docs.tropi-climate.id/data/satellite",
        },
    },
    {
        "name": "Auth",
        "description": (
            "JWT token issuance. For integration testing only — "
            "production systems use the OIDC provider at https://id.tropi-climate.id."
        ),
    },
    {
        "name": "Meta",
        "description": "Health checks and API capability discovery.",
    },
]

# ---------------------------------------------------------------------------
# x-partner-extensions (injected into the OpenAPI info object)
# ---------------------------------------------------------------------------

X_PARTNER_EXTENSIONS: dict[str, Any] = {
    "x-partner-documentation": {
        "bmkg": {
            "name": "Badan Meteorologi, Klimatologi, dan Geofisika",
            "integration_guide": "https://docs.tropi-climate.id/partners/bmkg",
            "rate_tier": "government",
        },
        "klhk": {
            "name": "Kementerian Lingkungan Hidup dan Kehutanan",
            "integration_guide": "https://docs.tropi-climate.id/partners/klhk",
            "rate_tier": "government",
        },
        "lapan": {
            "name": "Lembaga Penerbangan dan Antariksa Nasional",
            "integration_guide": "https://docs.tropi-climate.id/partners/lapan",
            "rate_tier": "government",
        },
    },
    "x-rate-limits": {
        "public": {"requests_per_hour": 100, "burst": 10},
        "researcher": {"requests_per_hour": 1000, "burst": 50},
        "government": {"requests_per_hour": 10000, "burst": 500},
        "admin": {"requests_per_hour": "unlimited"},
    },
    "x-data-freshness": {
        "climate_current": "≤ 5 minutes (Redis TTL 300s)",
        "forecast": "≤ 15 minutes (Redis TTL 900s)",
        "flood_alerts": "Real-time (Redis TTL 60s)",
        "streamflow": "≤ 30 seconds (Redis TTL 30s)",
        "satellite_scenes": "Per overpass (~16 days Landsat, ~1-2 days MODIS)",
    },
    "x-sdk-availability": {
        "python": "pip install tropi-climate-sdk",
        "javascript": "npm install @tropi-climate/sdk",
        "r": "devtools::install_github('KOSASIH/tropi-climate-analytics', subdir='sdks/r')",
    },
    "x-compliance": {
        "regulation": "PP Nomor 71 Tahun 2019",
        "data_residency": "Indonesia (AWS ap-southeast-3 Jakarta)",
        "audit_logging": true,
        "encryption_in_transit": "TLS 1.3",
        "encryption_at_rest": "AES-256",
    },
}

# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------


def build_custom_openapi(app: FastAPI) -> dict:
    """
    Generate a customised OpenAPI schema for the FastAPI app, injecting
    partner documentation extensions, tag descriptions, and contact info.

    Called once at startup in main.py::_patch_openapi().
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title="Tropi Climate Analytics API",
        version="1.0.0",
        description=OPENAPI_DESCRIPTION,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Inject partner extensions into the info block
    schema["info"].update(
        {
            "contact": PARTNER_CONTACT,
            "license": PARTNER_LICENSE,
            **X_PARTNER_EXTENSIONS,
        }
    )

    # External docs link
    schema["externalDocs"] = EXTERNAL_DOCS

    # Add Kong rate-limit response headers to all 200 responses
    rate_limit_headers = {
        "X-RateLimit-Limit": {
            "description": "Request limit per hour for this role tier",
            "schema": {"type": "integer"},
        },
        "X-RateLimit-Remaining": {
            "description": "Requests remaining in current window",
            "schema": {"type": "integer"},
        },
        "X-RateLimit-Reset": {
            "description": "Unix timestamp when the window resets",
            "schema": {"type": "integer"},
        },
        "X-Request-ID": {
            "description": "Correlation ID injected by Kong",
            "schema": {"type": "string"},
        },
    }

    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                for resp_code, resp in operation.get("responses", {}).items():
                    if str(resp_code).startswith("2"):
                        resp.setdefault("headers", {}).update(rate_limit_headers)

    return schema
