---
type: markdown
title: Developer Documentation Portal — Outline
---

# DEVELOPER DOCUMENTATION PORTAL
## Tropi Climate Analytics — Portal Content Outline
### COMM-ONNECT Sprint 0 | Coordinated with API-GATEWAY

---

## Portal Structure

```
docs.tropi-climate-analytics.id
├── Getting Started
│   ├── Platform Overview
│   ├── Quick Start (5 minutes)
│   ├── Authentication
│   └── Access Tiers & Rate Limits
├── API Reference
│   ├── OpenAPI 3.0 Specification
│   ├── Endpoints by Domain
│   │   ├── /v1/deforestation
│   │   ├── /v1/air-quality
│   │   ├── /v1/flood-warning
│   │   ├── /v1/precipitation
│   │   ├── /v1/sea-surface
│   │   └── /v1/alerts
│   ├── Request / Response Formats
│   ├── Error Codes
│   └── Changelog & Versioning
├── SDKs & Client Libraries
│   ├── Python SDK
│   ├── JavaScript/TypeScript SDK
│   ├── R Package
│   └── GIS Clients (QGIS, ArcGIS)
├── Data Products
│   ├── Product Catalogue
│   ├── Satellite Sources
│   ├── Ground Network Integration
│   ├── Data Formats (NetCDF, GeoTIFF, JSON, CSV)
│   ├── Spatial Resolution & Coverage
│   └── Data Quality & Validation
├── Tutorials & Examples
│   ├── Jupyter Notebooks
│   ├── Use Case Walkthroughs
│   └── Video Tutorials
├── Webhooks & Real-Time
│   ├── Webhook Setup
│   ├── Alert Types
│   └── Backendless Push Integration
├── Compliance & Data Policy
│   ├── Data License (CC-BY 4.0)
│   ├── PP 71/2019 Compliance
│   ├── Data Residency
│   └── Attribution Requirements
└── Support
    ├── FAQ
    ├── Helpdesk
    └── Community Forum
```

---

## Section 1: Getting Started

### 1.1 Platform Overview
*Target audience: all developers*

Brief explanation (3 paragraphs):
- What Tropi Climate Analytics is and what problems it solves
- Who the data is for: government agencies, researchers, private sector
- What makes it unique: NASA + Indonesian ground networks, open access, PP 71/2019 compliant

### 1.2 Quick Start (5 Minutes)
*Target audience: developers wanting immediate first call*

```python
# Example: Get latest AQI for Jakarta
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.tropi-climate-analytics.id/v1"

response = requests.get(
    f"{BASE_URL}/air-quality/current",
    params={"district_id": "3100", "fields": "aqi,pm25,pm10"},
    headers={"Authorization": f"Bearer {API_KEY}"}
)

data = response.json()
print(f"Jakarta AQI: {data['aqi']} — Category: {data['category']}")
```

Step-by-step:
1. Register at https://tropi-climate-analytics.id/developers
2. Generate API key in dashboard
3. Make your first call (above example)
4. Explore endpoints and SDKs

### 1.3 Authentication
*JWT Bearer token with RBAC tiers*

- **Public tier**: No auth required — current map tiles, district profiles
- **Standard tier**: API key (free registration) — current data, 30-day history
- **Research tier**: Institutional registration — full historical archive, bulk export
- **Government tier**: By agreement — real-time feeds, operational SLA, agency dashboards

Token format: `Authorization: Bearer <jwt_token>`
Token lifetime: 24 hours (auto-refresh endpoint available)

### 1.4 Access Tiers & Rate Limits

| Tier | Rate Limit | History | Bulk Export | SLA |
|---|---|---|---|---|
| Public | 100 req/hour | Current only | No | Best-effort |
| Standard | 1,000 req/hour | 30 days | Limited | Best-effort |
| Research | 10,000 req/hour | 20+ years | Full | 99% uptime |
| Government | Unlimited | Full | Full | 99.5% uptime |

---

## Section 2: API Reference

### 2.1 Endpoints by Domain

#### `/v1/deforestation`
| Endpoint | Method | Description |
|---|---|---|
| `/alerts/active` | GET | Current active deforestation alerts nationwide |
| `/alerts/{alert_id}` | GET | Alert detail with GeoJSON boundary |
| `/change-detection/weekly` | GET | Weekly change detection composites |
| `/change-detection/history` | GET | Historical change detection by province/district |
| `/hotspots` | GET | Active deforestation hotspot points (lat/lng) |

#### `/v1/air-quality`
| Endpoint | Method | Description |
|---|---|---|
| `/current` | GET | Current AQI by district (ISPU standard) |
| `/stations` | GET | List of integrated KLHK AQMS stations |
| `/forecast/24h` | GET | 24-hour AQI forecast |
| `/episodes/wildfire` | GET | Active wildfire smoke episodes |
| `/history` | GET | Historical AQI time series by district |

#### `/v1/flood-warning`
| Endpoint | Method | Description |
|---|---|---|
| `/alerts/active` | GET | Active flood early warnings |
| `/rivers` | GET | Monitored river basin list with current status |
| `/rivers/{basin_id}/forecast` | GET | Streamflow forecast (6-24h) for basin |
| `/inundation/current` | GET | Current flood inundation extent (GeoJSON) |
| `/rainfall/qpe` | GET | Quantitative Precipitation Estimate (GPM + BMKG) |

#### `/v1/precipitation`
| Endpoint | Method | Description |
|---|---|---|
| `/nowcast` | GET | 24-72 hour precipitation nowcast |
| `/seasonal` | GET | Seasonal (3-6 month) outlook |
| `/radar` | GET | Current radar composite (30-min update) |
| `/enso/current` | GET | Current ENSO index and Indonesia impact forecast |

#### `/v1/sea-surface`
| Endpoint | Method | Description |
|---|---|---|
| `/temperature/current` | GET | Current SST map (GeoTIFF or JSON) |
| `/temperature/anomaly` | GET | SST anomaly vs 20-year mean |
| `/marine-heatwave` | GET | Active marine heatwave alerts |
| `/chlorophyll` | GET | Chlorophyll-a concentration (fisheries) |

#### `/v1/alerts`
| Endpoint | Method | Description |
|---|---|---|
| `/active` | GET | All active alerts across all domains |
| `/subscribe` | POST | Register webhook for alert notifications |
| `/subscriptions` | GET | List active webhook subscriptions |

### 2.2 Common Parameters

| Parameter | Type | Description |
|---|---|---|
| `province_id` | string | BPS province code (e.g., "31" for DKI Jakarta) |
| `district_id` | string | BPS district/city code (e.g., "3100") |
| `lat` / `lng` | float | WGS84 coordinate lookup |
| `bbox` | string | Bounding box: "minLon,minLat,maxLon,maxLat" |
| `start_date` / `end_date` | string | ISO 8601 date range |
| `format` | string | Response format: `json` (default), `geojson`, `netcdf`, `geotiff` |
| `fields` | string | Comma-separated field selection |

---

## Section 3: SDKs & Client Libraries

### 3.1 Python SDK
```bash
pip install tropi-climate
```
Covers: all API endpoints, async support (httpx), GeoPandas integration, xarray for climate data, automatic token refresh.

### 3.2 JavaScript/TypeScript SDK
```bash
npm install @tropi/climate-sdk
```
Covers: all API endpoints, TypeScript types, React hooks for real-time data, Mapbox GL JS integration helper.

### 3.3 R Package
```r
install.packages("tropiclimate")
```
Covers: all API endpoints, sf spatial integration, raster/terra for gridded data, tidyverse-compatible output.

### 3.4 GIS Clients
- **QGIS Plugin**: Tropi Climate Analytics data catalog connection (WMS/WMTS + REST)
- **ArcGIS Connector**: ArcGIS Pro toolbox for direct platform data access

---

## Section 4: Data Products Catalogue

### 4.1 Satellite Sources Referenced

| Satellite | Agency | Resolution | Key Variables |
|---|---|---|---|
| Landsat 8/9 | NASA/USGS | 30m optical | Land cover, deforestation change |
| MODIS Terra/Aqua | NASA | 250m–1km | Vegetation, aerosol, SST, fire |
| VIIRS (Suomi-NPP) | NASA/NOAA | 375m | Active fire, night lights |
| GPM IMERG | NASA/JAXA | 0.1° | Precipitation rate |
| SMAP | NASA | 9km | Soil moisture |
| Himawari-9 | JMA | 2km | Cloud, moisture, convection |
| SPOT-7 | Airbus / LAPAN | 1.5m | High-res ground truth |
| SAR (L/C-band) | LAPAN/BRIN | 10–20m | Cloud-penetrating deforestation/flood |

---

## Section 5: Tutorials & Jupyter Notebooks

### Notebook Index

| Notebook | Level | Description |
|---|---|---|
| `01_quickstart.ipynb` | Beginner | First API call, plot AQI map of Indonesia |
| `02_deforestation_alerts.ipynb` | Intermediate | Weekly deforestation monitoring for Kalimantan |
| `03_flood_early_warning.ipynb` | Intermediate | Ciliwung basin streamflow + flood alert |
| `04_air_quality_timeseries.ipynb` | Intermediate | Jakarta PM2.5 trend analysis (5 years) |
| `05_sar_optical_fusion.ipynb` | Advanced | Cloud-penetrating deforestation mapping |
| `06_ml_precipitation_forecast.ipynb` | Advanced | Custom precipitation nowcast with XGBoost |
| `07_climate_indices.ipynb` | Intermediate | SPI/SPEI drought monitoring for all provinces |
| `08_satellite_time_lapse.ipynb` | Intermediate | Animated NDVI time-lapse for study area |

---

## Section 6: Webhooks & Real-Time Alerts

### Alert Types (8 types via Backendless push channels)
1. `flood` — Flood early warning for monitored river basins
2. `drought` — SPI/SPEI drought threshold exceedance
3. `aqi` — AQI threshold exceedance (ISPU Berbahaya/Sangat Tidak Sehat)
4. `deforestation` — Weekly deforestation alert exceeding district threshold
5. `heat` — Heat stress index threshold exceedance
6. `storm` — Severe convective storm detection
7. `wildfire` — Active fire VIIRS alert (district-level)
8. `sst_anomaly` — Sea surface temperature anomaly alert

### Webhook Payload Example
```json
{
  "alert_id": "def_2026_06_04_kaltim_001",
  "type": "deforestation",
  "severity": "HIGH",
  "district_id": "6472",
  "district_name": "Samarinda",
  "province": "Kalimantan Timur",
  "area_ha": 847.3,
  "coordinates": {"lat": -0.4948, "lng": 117.1436},
  "detected_at": "2026-06-04T02:30:00Z",
  "satellite_source": "Landsat-9",
  "confidence": 0.91
}
```

---

## Section 7: Compliance & Data Policy

### 7.1 Data License
All Tropi Climate Analytics output data products are released under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.

Required attribution format:
> "Data: Tropi Climate Analytics (tropi-climate-analytics.id), derived from NASA [Satellite] and [BMKG/KLHK/LAPAN] ground observations. License: CC-BY 4.0."

### 7.2 PP 71/2019 Compliance
All raw data is stored and processed exclusively on **AWS ap-southeast-3 (Jakarta)**, complying with PP No. 71/2019 on Electronic System and Transaction Implementation (Peraturan Pemerintah tentang Penyelenggaraan Sistem dan Transaksi Elektronik), which mandates Indonesian data residency for strategic electronic data.

### 7.3 Partner Data Usage
BMKG, KLHK, and LAPAN raw observation data is used under individual MOU agreements with each agency. These MOUs restrict raw data redistribution; only processed/derived products (AQI, deforestation alerts, flood forecasts) are available via the public API.
