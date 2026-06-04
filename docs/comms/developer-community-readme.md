# Tropi Climate Analytics

<div align="center">

<img src="https://tropi-climate-analytics.id/assets/logo-banner.png" alt="Tropi Climate Analytics" width="600"/>

**Indonesia's Integrated Satellite & Ground Climate Intelligence Platform**

[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightblue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Data Residency: Indonesia](https://img.shields.io/badge/Data%20Residency-Indonesia%20(PP%2071%2F2019)-green.svg)](https://peraturan.bpk.go.id/Home/Details/110486)
[![API Status](https://img.shields.io/badge/API-Live-brightgreen.svg)](https://api.tropi-climate-analytics.id/health)
[![Docs](https://img.shields.io/badge/Docs-docs.tropi--climate--analytics.id-blue.svg)](https://docs.tropi-climate-analytics.id)
[![Python SDK](https://img.shields.io/pypi/v/tropi-climate.svg?label=tropi-climate)](https://pypi.org/project/tropi-climate/)
[![GitHub Stars](https://img.shields.io/github/stars/KOSASIH/tropi-climate-analytics?style=social)](https://github.com/KOSASIH/tropi-climate-analytics/stargazers)

*NASA · BMKG · KLHK · LAPAN/BRIN — Real-time. Open access. CC-BY 4.0.*

[Platform](https://tropi-climate-analytics.id) · [API Docs](https://docs.tropi-climate-analytics.id) · [Notebooks](./notebooks/) · [Contributing](#contributing) · [Community](#community)

</div>

---

## What is Tropi Climate Analytics?

Tropi Climate Analytics is Indonesia's first cloud-native climate intelligence platform, integrating NASA Earth observation satellite data with national ground observation networks from BMKG, KLHK, and LAPAN into a unified, open-access analytics infrastructure.

The platform delivers real-time monitoring across five critical domains for all 34 Indonesian provinces and 514 districts:

| Domain | Satellite Sources | Ground Sources | Update Frequency |
|--------|------------------|----------------|-----------------|
| 🌿 **Deforestation** | Landsat 8/9 (30m), MODIS, VIIRS, SAR (LAPAN) | KLHK land cover maps, SPOT-7 validation | Weekly alerts |
| 💨 **Air Quality** | MODIS MAIAC aerosol, VIIRS fire | 147 KLHK AQMS stations | Hourly |
| 🌊 **Flood Early Warning** | GPM IMERG (0.1°), SMAP soil moisture | BMKG rain gauges, river gauges | Every 30 min |
| 🌧️ **Precipitation** | GPM, Himawari-9, BMKG radar | BMKG AWS network (500+ stations) | Sub-hourly |
| 🌡️ **Sea Surface Temp** | MODIS SST, VIIRS | BMKG buoy network | Daily |

**All outputs: CC-BY 4.0. Data stored in Indonesia (AWS ap-southeast-3) per PP No. 71/2019.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TROPI CLIMATE ANALYTICS                          │
│                    Platform Architecture                            │
├───────────────────┬─────────────────────┬───────────────────────────┤
│   DATA SOURCES    │   PROCESSING        │   OUTPUTS                 │
│                   │                     │                           │
│  NASA Satellites  │  ┌───────────────┐  │  🌐 Web Platform          │
│  ─────────────    │  │  DATA-FLOW    │  │  https://tropi-           │
│  • Landsat 8/9    │  │  Ingestion &  │  │  climate-analytics.id     │
│  • MODIS Terra/   │→ │  ETL Pipeline │→ │                           │
│    Aqua           │  └───────────────┘  │  🔑 REST API              │
│  • VIIRS          │         ↓           │  api.tropi-               │
│  • GPM IMERG      │  ┌───────────────┐  │  climate-analytics.id     │
│  • SMAP           │  │   PostGIS +   │  │                           │
│                   │  │   Feature     │  │  📓 Python / R / JS       │
│  BMKG Network     │  │   Store       │  │  SDKs                     │
│  ─────────────    │  └───────────────┘  │                           │
│  • 500+ AWS       │         ↓           │  📊 Dashboard             │
│  • Radar          │  ┌───────────────┐  │  (TERRA-VISION)           │
│  • Rain gauges    │  │  ANALYTICA    │  │                           │
│                   │  │  ML Models    │  │  🔔 Webhooks              │
│  KLHK Network     │  │  (XGBoost,    │  │  Real-time alerts via     │
│  ─────────────    │  │  Prophet,     │  │  FCM / APNs               │
│  • 147 AQMS       │  │  CNN, SWAT)   │  │  (Backendless push)       │
│  • Forest maps    │  └───────────────┘  │                           │
│                   │         ↓           │  📥 Bulk Data Export      │
│  LAPAN/BRIN       │  ┌───────────────┐  │  NetCDF, GeoTIFF,         │
│  ─────────────    │  │  API-GATEWAY  │  │  JSON, CSV                │
│  • SAR data       │  │  (Kong + Fast │  │                           │
│  • SPOT-7         │  │  API + Redis) │  │                           │
│  • Himawari-9     │  └───────────────┘  │                           │
└───────────────────┴─────────────────────┴───────────────────────────┘
                         ↑           ↑
               AWS ap-southeast-3  (Jakarta)
               PP No. 71/2019 compliant data residency
```

*Architecture diagram (visual): [`docs/architecture/platform-architecture.png`](./docs/architecture/) — placeholder, to be rendered by VISUALIA.*

---

## Quick Start

### 1. Get an API Key

Register at **https://tropi-climate-analytics.id/developers** — free for researchers and government agencies.

### 2. Install the Python SDK

```bash
pip install tropi-climate
```

### 3. Make Your First API Call

```python
from tropiclient import TropiClient

# Initialize with your API key
client = TropiClient(api_key="your_api_key_here")

# Get current AQI for Jakarta (district code: 3100)
aqi = client.air_quality.current(district_id="3100")
print(f"Jakarta AQI: {aqi['aqi']} — Category: {aqi['category']}")
# Jakarta AQI: 87 — Category: Moderate (ISPU)

# Get active deforestation alerts (nationwide)
alerts = client.deforestation.alerts.active()
print(f"Active deforestation alerts: {len(alerts)} events")
for alert in alerts[:3]:
    print(f"  → {alert['district_name']}: {alert['area_ha']:.1f} ha ({alert['severity']})")

# Get flood early warning status for Ciliwung basin
flood = client.flood_warning.rivers.forecast(basin_id="ciliwung")
print(f"Ciliwung forecast (6h): {flood['status']} — Peak flow: {flood['peak_flow_m3s']} m³/s")
```

**Output:**
```
Jakarta AQI: 87 — Category: Moderate (ISPU)
Active deforestation alerts: 12 events
  → Kutai Timur: 847.3 ha (HIGH)
  → Berau: 423.1 ha (MEDIUM)
  → Barito Utara: 216.8 ha (MEDIUM)
Ciliwung forecast (6h): ELEVATED — Peak flow: 342.7 m³/s
```

### 4. Explore More

| Resource | Link |
|----------|------|
| 📖 Full API documentation | https://docs.tropi-climate-analytics.id |
| 📓 Jupyter notebooks (8 examples) | [`./notebooks/`](./notebooks/) |
| 🐍 Python SDK reference | https://docs.tropi-climate-analytics.id/python-sdk |
| 📊 R package | https://docs.tropi-climate-analytics.id/r-package |
| 🟨 JavaScript/TypeScript SDK | https://docs.tropi-climate-analytics.id/js-sdk |

---

## Jupyter Notebooks

Ready-to-run examples in [`./notebooks/`](./notebooks/):

| Notebook | Level | Description |
|----------|-------|-------------|
| [`01_quickstart.ipynb`](./notebooks/01_quickstart.ipynb) | Beginner | First API call — Indonesia-wide AQI map |
| [`02_deforestation_alerts.ipynb`](./notebooks/02_deforestation_alerts.ipynb) | Intermediate | Weekly deforestation monitoring for Kalimantan |
| [`03_flood_early_warning.ipynb`](./notebooks/03_flood_early_warning.ipynb) | Intermediate | Ciliwung basin streamflow + alert visualization |
| [`04_air_quality_timeseries.ipynb`](./notebooks/04_air_quality_timeseries.ipynb) | Intermediate | Jakarta PM2.5 trend analysis — 5 years |
| [`05_sar_optical_fusion.ipynb`](./notebooks/05_sar_optical_fusion.ipynb) | Advanced | All-weather deforestation mapping (SAR + Landsat) |
| [`06_ml_precipitation_forecast.ipynb`](./notebooks/06_ml_precipitation_forecast.ipynb) | Advanced | Custom precipitation nowcast with XGBoost |
| [`07_climate_indices.ipynb`](./notebooks/07_climate_indices.ipynb) | Intermediate | SPI/SPEI drought monitoring — all provinces |
| [`08_satellite_time_lapse.ipynb`](./notebooks/08_satellite_time_lapse.ipynb) | Intermediate | Animated NDVI time-lapse for a study area |

All notebooks are runnable in **Google Colab** without local setup. Click the badge at the top of each notebook to launch.

---

## API Reference

Base URL: `https://api.tropi-climate-analytics.id/v1`

| Endpoint Group | Description | Docs |
|----------------|-------------|------|
| `/deforestation` | Weekly change detection, hotspots, alerts | [→](https://docs.tropi-climate-analytics.id/api/deforestation) |
| `/air-quality` | AQI, PM2.5, AQMS stations, wildfire smoke | [→](https://docs.tropi-climate-analytics.id/api/air-quality) |
| `/flood-warning` | River forecasts, flood alerts, inundation extents | [→](https://docs.tropi-climate-analytics.id/api/flood-warning) |
| `/precipitation` | Nowcasts, seasonal outlooks, radar, ENSO | [→](https://docs.tropi-climate-analytics.id/api/precipitation) |
| `/sea-surface` | SST, SST anomaly, marine heatwave, chlorophyll | [→](https://docs.tropi-climate-analytics.id/api/sea-surface) |
| `/alerts` | Active alerts (all domains), webhook subscriptions | [→](https://docs.tropi-climate-analytics.id/api/alerts) |

**Authentication:** `Authorization: Bearer <your_jwt_token>`

**Rate limits by tier:**

| Tier | Rate | History | Registration |
|------|------|---------|--------------|
| Standard (free) | 1,000 req/hr | 30 days | Self-service |
| Research | 10,000 req/hr | 20+ years | Institutional |
| Government | Unlimited | Full | By agreement |

---

## Data License

All platform output data products are released under **[Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

**Required attribution:**
> "Data: Tropi Climate Analytics (tropi-climate-analytics.id), derived from NASA [satellite] and [BMKG/KLHK/LAPAN] ground observations. License: CC-BY 4.0."

**Data residency:** All raw data is stored and processed exclusively on **AWS ap-southeast-3 (Jakarta)**, complying with **PP No. 71/2019** on Indonesian data residency.

---

## Institutional Partners

<div align="center">

| Partner | Role | Data Provided |
|---------|------|---------------|
| [BMKG](https://www.bmkg.go.id) | Primary weather data partner | AWS observations, radar, rain gauges, seasonal forecasts |
| [KLHK](https://www.menlhk.go.id) | Forest & air quality partner | Land cover maps, AQMS stations, forest permits, fire data |
| [LAPAN/BRIN](https://lapan.go.id) | Remote sensing partner | SAR data, SPOT-7, Himawari-9, LAPAN-A series |
| [NASA](https://earthdata.nasa.gov) | Satellite data | Landsat, MODIS, SMAP, GPM, VIIRS, GRACE-FO |

</div>

---

## Repository Structure

```
tropi-climate-analytics/
├── docs/
│   ├── comms/              # COMM-ONNECT: engagement letters, press releases, social media
│   ├── architecture/       # Platform architecture diagrams
│   ├── api/                # API reference documentation
│   └── partnerships/       # MOU documents and partnership frameworks
├── notebooks/              # Jupyter notebooks (8 analysis examples)
├── src/
│   ├── dashboard/          # TERRA-VISION: React/TypeScript dashboard components
│   ├── push-pipeline/      # Real-time alert push pipeline (Backendless FCM/APNs)
│   └── data/               # Province/district reference data (provinces.json)
├── config/
│   └── superset/           # Apache Superset dashboard configuration (7 widgets)
├── sdk/
│   ├── python/             # Python SDK (tropi-climate)
│   ├── javascript/         # JavaScript/TypeScript SDK (@tropi/climate-sdk)
│   └── r/                  # R package (tropiclimate)
└── README.md
```

---

## Contributing

We welcome contributions from the Indonesian and international climate research community.

### Ways to Contribute

1. **Report data quality issues** → [Open an issue](https://github.com/KOSASIH/tropi-climate-analytics/issues) with tag `data-quality`
2. **Contribute Jupyter notebooks** → Submit a PR to `./notebooks/` with a new analysis example
3. **Improve documentation** → PRs to `./docs/` always welcome
4. **Report bugs** → [Open an issue](https://github.com/KOSASIH/tropi-climate-analytics/issues) with tag `bug`
5. **Suggest new data products** → [Open an issue](https://github.com/KOSASIH/tropi-climate-analytics/issues) with tag `feature-request`

### Contribution Guidelines

1. **Fork** this repository and create your branch from `main`
2. **Write clear commit messages** — use Conventional Commits format: `feat(scope): description`
3. **Document your changes** — update relevant docs in `./docs/`
4. **For notebooks** — ensure the notebook runs end-to-end on Google Colab before submitting
5. **Open a pull request** — describe what you changed and why

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.

---

## Community

| Channel | Link | Purpose |
|---------|------|---------|
| 🌐 Platform | https://tropi-climate-analytics.id | Access data and dashboards |
| 📖 Docs | https://docs.tropi-climate-analytics.id | API docs and developer guides |
| 🐦 Twitter/X | [@TropiClimate](https://twitter.com/TropiClimate) | Daily climate alerts and updates |
| 💼 LinkedIn | [Tropi Climate Analytics](https://linkedin.com/company/tropi-climate-analytics) | Platform news and research highlights |
| 📺 YouTube | [TropiClimateAnalytics](https://youtube.com/@TropiClimateAnalytics) | Webinar recordings and tutorials |
| 📧 General | info@tropi-climate-analytics.id | General inquiries |
| 🤝 Partnerships | partnerships@tropi-climate-analytics.id | Agency and research partnerships |
| 🔑 API Support | api-support@tropi-climate-analytics.id | Developer support |
| 📰 Press | press@tropi-climate-analytics.id | Media inquiries |

---

## Citation

If you use Tropi Climate Analytics data in a publication, please cite:

```bibtex
@misc{tropi2026,
  author       = {Tropi Climate Analytics},
  title        = {Tropi Climate Analytics: Indonesia's Integrated Satellite-Ground Climate Intelligence Platform},
  year         = {2026},
  howpublished = {\url{https://tropi-climate-analytics.id}},
  note         = {Data: NASA [Landsat/MODIS/SMAP/GPM/VIIRS] + BMKG + KLHK + LAPAN. License: CC-BY 4.0}
}
```

---

<div align="center">

*Built for Indonesia. Open to the world.*

**[tropi-climate-analytics.id](https://tropi-climate-analytics.id)**

[![Twitter Follow](https://img.shields.io/twitter/follow/TropiClimate?style=social)](https://twitter.com/TropiClimate)

</div>
