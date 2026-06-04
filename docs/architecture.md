# Tropi-Climate-Analytics Platform Architecture

## System Overview

The platform is organized around 11 AI agents covering the full data lifecycle.

## Data Flow

```
NASA Earthdata → CMR API → Data Ingestion → Kafka → ETL → PostGIS/S3 → API → Dashboard
BMKG/KLHK/LAPAN → REST APIs → Data Ingestion → Kafka → ETL → PostGIS/S3 → API → Dashboard
```

## Services

| Service | Technology | Agent | Port |
|---------|-----------|-------|------|
| API Gateway | FastAPI + Kong | API-GATEWAY | 8000 |
| Message Queue | Apache Kafka | DATA-FLOW | 9092 |
| Workflow Orchestration | Apache Airflow | DATA-FLOW | 8080 |
| Spatial Database | PostgreSQL + PostGIS | GEOSPATIAL | 5432 |
| Cache | Redis | API-GATEWAY | 6379 |
| ML Tracking | MLflow | ANALYTICA | 5000 |
| Monitoring | Prometheus + Grafana | CLOUD-FORGE | 9090/3000 |
| Dashboard | Apache Superset | TERRA-VISION | 8088 |

## Coordinate Reference Systems

- **WGS84 (EPSG:4326)** — Default for API responses and storage
- **DGN-95 (EPSG:23845–23847)** — Indonesian national datum for official outputs
- **UTM Zone 46–54S/N** — For local analysis and area calculations

## Security Layers

1. Kong API Gateway — rate limiting, JWT validation
2. AWS WAF — DDoS protection, IP allowlist
3. VPC segmentation — public/private subnets
4. TLS 1.3 — all external traffic
5. AWS IAM — least-privilege service roles

## Disaster Recovery

- RTO: 4 hours | RPO: 1 hour
- Multi-AZ PostgreSQL with automated failover
- S3 cross-region replication for raw data
- Daily database snapshots retained 30 days
