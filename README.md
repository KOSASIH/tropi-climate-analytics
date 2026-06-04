# 🌍 Tropi-Climate-Analytics

> **Cloud-based big-data analytics platform for tropical climate monitoring**
> Integrates NASA satellite data (Landsat, MODIS, SMAP, GPM) with Indonesian ground observations from BMKG, KLHK, and LAPAN.

[![CI/CD](https://github.com/KOSASIH/tropi-climate-analytics/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/KOSASIH/tropi-climate-analytics/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Agents](https://img.shields.io/badge/AI%20Agents-11-brightgreen)](docs/agents.md)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│  DATA SOURCES: NASA Landsat/MODIS/SMAP/GPM + BMKG/KLHK/LAPAN │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  DATA INGESTION: Apache Kafka Streaming + Airflow Batch  │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  PROCESSING: ETL Pipelines │ ML Models │ Geospatial      │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  STORAGE: PostgreSQL/PostGIS │ Redis │ AWS S3            │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  API GATEWAY: FastAPI + Kong │ JWT Auth │ Rate Limiting  │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  VISUALIZATION: Dashboard │ Maps (Mapbox) │ Alerts       │
└──────────────────────────────────────────────────────────┘
```

## 📁 Structure

```
tropi-climate-analytics/
├── .github/workflows/       # CI/CD pipelines
├── docs/                    # Architecture, data sources, partnerships
├── src/
│   ├── api/                 # FastAPI application + routes
│   ├── data/
│   │   ├── ingestion/       # NASA CMR + BMKG/KLHK/LAPAN clients
│   │   ├── processing/      # ETL pipelines
│   │   └── storage/         # Database & cache layer
│   ├── analytics/           # ML models (XGBoost, Prophet, CNN)
│   ├── visualization/       # Dashboard components
│   └── tests/               # Unit & integration tests
├── scripts/                 # Deployment & setup scripts
├── config/                  # Application configuration
├── docker/                  # Container definitions
├── notebooks/               # Jupyter analysis notebooks
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## 🚀 Quick Start

```bash
git clone https://github.com/KOSASIH/tropi-climate-analytics.git
cd tropi-climate-analytics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys
python scripts/setup/initialize_db.py
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API Docs: http://localhost:8000/docs

## 🤖 AI Agent Ecosystem (11 Agents)

| Agent | Role |
|-------|------|
| **CLIMATE-OS** | Project Director & Executive Orchestrator |
| **DATA-FLOW** | NASA/BMKG Data Pipeline & Kafka Streaming |
| **ATMOSPHERE** | MODIS/GPM Processing, AQI, Weather Alerts |
| **GEOSPATIAL** | PostGIS, Landsat, Change Detection, WMS |
| **HYDROLOGIS** | SWAT Model, Flood Prediction, SMAP Soil Moisture |
| **ANALYTICA** | ML/AI Models, MLflow, Predictive Analytics |
| **TERRA-VISION** | Dashboard, Maps, Real-time Visualization |
| **API-GATEWAY** | Kong Gateway, FastAPI, JWT Auth, SDKs |
| **SECURE-GUARD** | IAM, WAF, CloudTrail, Security Compliance |
| **CLOUD-FORGE** | Terraform, Kubernetes, Prometheus/Grafana |
| **COMM-ONNECT** | Stakeholders, Media, BMKG/KLHK/LAPAN Relations |

## 🌐 Data Sources

| Source | Product | Resolution | Update |
|--------|---------|-----------|--------|
| NASA Landsat 8/9 | Land cover, deforestation | 30m | 16-day |
| NASA MODIS | NDVI, fire, aerosols, SST | 250m–1km | Daily |
| NASA SMAP | Soil moisture | 9km | Daily |
| NASA GPM IMERG | Precipitation | 0.1° | 30-min |
| BMKG | Surface weather observations | Station | Hourly |
| KLHK | Forest cover change | 30m | Annual |
| LAPAN | Indonesian remote sensing | Various | Various |

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

---
*Built with ❤️ for Indonesia's climate resilience | CLIMATE-OS Platform v0.1.0*
