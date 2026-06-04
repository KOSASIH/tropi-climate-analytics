# HYDROLOGIS Router — model_server.py Integration Guide
**Sprint 4 | Deliverable 4**  
*Target base SHA: ANALYTICA `c418a904` / updated `20889a60`*

---

## Overview

This document specifies the **exact lines** ANALYTICA must add to `model_server.py` to wire in the HYDROLOGIS FastAPI router (`src/api/hydrologis_router.py`, Sprint 3 SHA `843892ca`).

---

## Integration Instructions

### 1. Import statement

Add **after** existing router imports (near the top of `model_server.py`):

```python
from src.api.hydrologis_router import router as hydrologis_router
```

### 2. Router registration

Add **after** other `app.include_router(...)` calls:

```python
app.include_router(hydrologis_router)
```

### Complete diff (context)

```diff
 # model_server.py
 from fastapi import FastAPI
 from src.api.climate_router import router as climate_router
+from src.api.hydrologis_router import router as hydrologis_router

 app = FastAPI(
     title="Tropi Climate Analytics — Model Server",
     version="0.4.0",
 )

 app.include_router(climate_router)
+app.include_router(hydrologis_router)
```

---

## Endpoints registered

Once wired, the following endpoints are live under prefix `/api/v1/hydrologis`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/hydrologis/streamflow/{river_id}/forecast` | Latest LSTM+HBV streamflow forecast (6/12/24h) |
| `GET` | `/api/v1/hydrologis/flood-extent/{river_id}` | Latest GeoJSON flood inundation polygon |
| `GET` | `/api/v1/hydrologis/agri-advisory/{watershed_id}` | Latest seasonal irrigation advisory |
| `GET` | `/api/v1/hydrologis/aquifer/{aquifer_id}/latest` | Latest GRACE-FO aquifer depletion report |
| `GET` | `/api/v1/hydrologis/health` | Health check (not in OpenAPI schema) |

All endpoints:
- Return `Cache-Control: public, max-age=1800`
- Return structured `404` with `error / message / hint` fields when output file not found
- Increment `tropi_hydrologis_api_requests_total{endpoint, status_code}` Prometheus counter on every response

---

## Prometheus registry — no collision

`hydrologis_router.py` imports `_REGISTRY` from `src.hydrology.metrics`:

```python
from src.hydrology.metrics import _REGISTRY
```

`metrics.py` creates a **single isolated `CollectorRegistry`** (not the `prometheus_client` default registry):

```python
_REGISTRY = CollectorRegistry()
```

All HYDROLOGIS metrics (Sprint 2 + Sprint 3 + Sprint 4) share this one registry. There is **no duplicate metric registration** and **no collision** with ANALYTICA's own metrics, which use either `prometheus_client.REGISTRY` (the default) or their own isolated registry.

---

## OpenAPI

After wiring, the HYDROLOGIS endpoints appear in the auto-generated Swagger UI at `/docs` under the `hydrologis` tag.

---

## Valid parameter values

| Endpoint | Parameter | Valid values |
|----------|-----------|--------------|
| `/streamflow/{river_id}/forecast` | `river_id` | `ciliwung` · `brantas` · `solo` |
| `/flood-extent/{river_id}` | `river_id` | `ciliwung` · `brantas` · `solo` |
| `/flood-extent/{river_id}` | `horizon_hours` *(query, optional)* | `6` · `12` · `24` (default: 24) |
| `/agri-advisory/{watershed_id}` | `watershed_id` | `das_ciliwung` · `das_citarum` · `das_serayu` · `das_bengawan_solo` · `das_brantas` · `das_musi` · `das_batang_hari` · `das_kapuas` · `das_barito` · `das_mahakam` · `das_jeneberang` · `das_saddang` · `das_memberamo` · `das_digul` · `das_progo_opak` · `das_pemali_comal` · `das_toba_asahan` · `das_cisanggarung` · `das_kampar` · `das_akucem` · `das_serayu` |
| `/aquifer/{aquifer_id}/latest` | `aquifer_id` | `north_jakarta` · `bandung_basin` · `semarang` · `surabaya` · `makassar` |

---

## Output file dependencies

Endpoints read from `workspace/output/`. Ensure the following pipelines have run at least once before calling these endpoints in production:

| Endpoint | Populated by DAG |
|----------|-----------------|
| `/streamflow/…` | `flood_early_warning_30min` |
| `/flood-extent/…` | `flood_early_warning_30min` |
| `/agri-advisory/…` | `hydrologis_agri_advisory` |
| `/aquifer/…` | `hydrologis_aquifer_tracker` |

---

*HYDROLOGIS Sprint 4 | Tropi Climate Analytics*  
*File SHA: to be assigned by CLIMATE-OS on commit*
