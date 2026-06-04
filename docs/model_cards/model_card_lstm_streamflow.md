---
model_id: lstm_streamflow
version: "1.2"
last_updated: "2026-06-04"
compliance_status: PP71_2019_COMPLIANT
contact: kosasihg88@gmail.com
---

# Model Card — LSTM Streamflow Forecast
**Model ID:** `lstm_streamflow` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## (a) Model Overview & Intended Use
Stacked 3-layer bidirectional LSTM for 6–72 hour river streamflow and flood stage forecasting at 23 BMKG AWLR telemetry stations across 12 priority Indonesian watersheds. Fuses HYDROLOGIS QPE (blended GPM+BMKG precipitation), SMAP soil moisture, and real-time AWLR river stage observations.

| Use Case | Status |
|---|---|
| 6–72 h streamflow / flood stage forecast | ✅ Supported |
| Flood early warning (HYDROLOGIS pipeline) | ✅ Supported |
| Dam/reservoir inflow forecasting (BBWS) | ✅ Supported |
| Tidal/coastal backwater flooding | ❌ Not supported |
| Watersheds outside training domain | ❌ Not supported |
| Groundwater level forecasting | ❌ Not supported |

**End users:** HYDROLOGIS agent (flood alert generation), BNPB disaster preparedness, PU/BBWS dam operations, BMKG flood desk.

---

## (b) Training Data
**Spatial coverage:** 12 priority watersheds, 23 BMKG AWLR stations, all within Indonesian territory.

**Covered watersheds:** Ciliwung-Cisadane (DKI Jakarta/West Java), Brantas (East Java), Solo (Central/East Java), Citarum (West Java), Mahakam (East Kalimantan), Barito (Central/South Kalimantan), Kapuas (West Kalimantan), Musi (South Sumatra), Kampar (Riau), Batanghari (Jambi), Rokan (Riau/North Sumatra), Membramo (Papua).

**Temporal range:** January 2015 – December 2025

| Source | Variable | Resolution | Period |
|---|---|---|---|
| HYDROLOGIS QPE (GPM IMERG + BMKG blend) | Catchment-averaged rainfall mm/h | 4 km / 30 min | 2015–2025 |
| SMAP L3 Enhanced (NASA) | Root-zone soil moisture m³/m³ | 9 km / 3-day | 2015–2025 |
| BMKG AWLR telemetry (23 stations) | River stage m, discharge m³/s | Station / 15 min | 2015–2025 |
| SRTM 30 m (NASA) | Catchment area, mean slope, TWI | Static | — |
| GEOSPATIAL annual land cover | Imperviousness fraction per sub-basin | 30 m / annual | 2015–2025 |

**Train / Val / Test split:** 2015–2022 / 2023 / 2024 (chronological).

---

## (c) Performance Metrics (2024 held-out test set)

### Streamflow (discharge m³/s)
| Lead Time | RMSE (m³/s) | MAE (m³/s) | NSE | KGE | PBIAS |
|---|---|---|---|---|---|
| 6 h | 84 | 51 | 0.91 | 0.89 | −3.1% |
| 24 h | 142 | 89 | 0.86 | 0.83 | −5.4% |
| 48 h | 213 | 138 | 0.78 | 0.75 | −8.2% |
| 72 h | 291 | 192 | 0.71 | 0.68 | −11.7% |

### Flood Event Detection (alert-stage threshold crossing)
| Metric | Value |
|---|---|
| POD (Probability of Detection) | 0.884 |
| FAR (False Alarm Rate) | 0.107 |
| CSI (Critical Success Index) | 0.806 |
| Median lead time before stage peak | 14.2 h |

---

## (d) Known Limitations & Failure Modes
- **AWLR telemetry outages:** Station gaps > 3 h significantly degrade forecast quality; system falls back to climatological stage prior with explicit uncertainty widening.
- **Extreme events beyond training envelope:** Events exceeding the 2013 Ciliwung and 2021 Kalimantan flood magnitudes fall outside training distribution; model is expected to underestimate peak discharge.
- **Dam/gate operations:** Not calibrated for dynamic dam gate scheduling; requires BBWS real-time gate telemetry integration (roadmap item Q3 2026).
- **Land-use change lag:** Annual imperviousness updates may trail actual urbanization in fast-developing peri-urban catchments by 6–12 months.
- **Tidal backwater:** Model does not account for tidal influence on downstream stage — affects lower Ciliwung and Brantas deltas.

---

## (e) Regional Fairness Analysis (24 h RMSE m³/s)
| Region | RMSE (m³/s) | Notes |
|---|---|---|
| Sumatra | 158 | Equatorial rainfall variability; adequate AWLR coverage |
| Jawa | 121 | Best; densest AWLR + BMKG network, most training data |
| Kalimantan | 174 | Large catchment areas; upstream telemetry gaps |
| Sulawesi | — | Not yet in training domain; planned Q4 2026 |
| Maluku + Papua | 231 | Highest uncertainty; Membramo sparse stations + extreme catchment size |

**Gap analysis:** Maluku+Papua (Membramo) shows 91% higher RMSE than Jawa. Flood alerts for Membramo basin must include explicit probabilistic uncertainty bounds in HYDROLOGIS alert output.

---

## (f) PP 71/2019 Data Residency Compliance
All telemetry data, model artifacts, and forecast outputs processed and stored within Indonesian jurisdiction on **AWS ap-southeast-3 (Jakarta)**. No cross-border data transfer without BSSN authorization per PP 71/2019 Article 17. BMKG AWLR telemetry under bilateral MoU (renewed 2024-09-01). Flood forecast outputs classified as *peringatan dini banjir* (early warning information) under PP 71/2019 Article 8; dissemination to the public requires BMKG operational validation and BNPB co-release authorization. No PII processed. Flood alert pipeline independently audited by BNPB (ref: BNPB-2025-AUDIT-042).

**Compliance status:** ✅ PP71_2019_COMPLIANT

---

## (g) Retraining Schedule
| Activity | Frequency |
|---|---|
| Drift check (PSI) | Daily — model_serving_health_check DAG |
| **Scheduled retraining** | **Monthly** (1st of each month, 04:00 WIB — incorporates new AWLR + SMAP data) |
| Emergency retrain | Automated on PSI CRITICAL |
| BBWS / BMKG accuracy review | Quarterly |

MLflow experiment: `lstm_streamflow` | Registry alias: `tropi_lstm_streamflow` | DAG: `model_serving_health_check`
