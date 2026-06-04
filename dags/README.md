# ============================================================
# Airflow DAGs — Tropi Climate Analytics
# HYDROLOGIS Hydrological Pipeline Definitions
# Managed by: CLOUD-FORGE (infra) + HYDROLOGIS (domain logic)
# ============================================================

## DAG Inventory

| DAG ID | Schedule | Trigger | Pipeline |
|---|---|---|---|
| `qpe_fusion_30min` | Every 30 min | `*/30 * * * *` | GPM IMERG-HHR + BMKG Kriging fusion |
| `flood_early_warning_30min` | Every 30 min | `*/30 * * * *` | LSTM streamflow forecast (Ciliwung, Brantas, Solo) |
| `smap_daily` | Daily 06:00 WIB | `0 23 * * *` (UTC) | SMAP L3 soil moisture ingestion |
| `grace_monthly` | 1st of month 03:00 WIB | `0 20 1 * *` (UTC) | GRACE-FO groundwater anomaly |
| `seasonal_monthly` | 1st of month 04:00 WIB | `0 21 1 * *` (UTC) | Water Availability Index |

## Pipeline Module Mapping

```
src/hydrology/
  qpe/gpm_bmkg_fusion.py       → GPMBMKGFusionPipeline
  flood/                        → FloodEarlyWarningPipeline
  soil_moisture/                → SMAPIngestionPipeline
  groundwater/                  → GRACEFOGroundwaterPipeline
  seasonal/                     → SeasonalWaterAvailabilityPipeline
```

## Timezone
All cron schedules are UTC. All WIB display times assume UTC+7.
