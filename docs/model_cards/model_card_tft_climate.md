---
model_id: tft_climate_forecast
version: "1.0"
last_updated: "2026-06-04"
compliance_status: PP71_2019_COMPLIANT
contact: kosasihg88@gmail.com
---

# Model Card — TFT Multi-Variate Climate Forecast
**Model ID:** `tft_climate_forecast` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## (a) Model Overview & Intended Use
Temporal Fusion Transformer (TFT) for multi-variate 7-day ahead climate forecasting of surface temperature (T2M) and relative humidity (RH) at province level across all 38 Indonesian provinces. Architecture: encoder context window 30 days, prediction horizon 7 days, hidden dim 256, 4 attention heads, LSTM encoder-decoder backbone. Registered as `tropi_climate_tft` in MLflow.

| Use Case | Status |
|---|---|
| 7-day T2M forecast (°C) at province level | ✅ Supported |
| 7-day RH forecast (%) at province level | ✅ Supported |
| Heat stress index input (VISUALIA / API-GATEWAY) | ✅ Supported |
| Seasonal (>7 day) multi-variate forecast | ❌ Not supported |
| Wind, pressure, or precipitation as primary output | ❌ Not supported |
| Sub-province spatial resolution | ❌ Not supported |

**End users:** VISUALIA dashboard heat-stress layers, API-GATEWAY partner data feeds, BMKG medium-range desk, Kemenkes (public health heat advisory).

---

## (b) Training Data
**Spatial coverage:** All 38 Indonesian provinces.

**Temporal range:** January 2015 – December 2025

| Source | Variable | Resolution | Period |
|---|---|---|---|
| ERA5 reanalysis (ECMWF) | T2M (°C), RH (%), U/V 10m winds, MSLP | 0.25° / hourly → daily | 2015–2025 |
| BMKG surface synoptic stations (210 stations) | T2M, Tmin, Tmax, RH, cloud cover, sunshine hours | Station / daily | 2015–2025 |
| MODIS MOD11A1 V6.1 | Land surface temperature K | 1 km / daily → province mean | 2015–2025 |
| NOAA CPC | ONI ENSO index | Monthly (interpolated daily) | 2015–2025 |
| GPM IMERG Late (NASA) | Precipitation mm/h (static regressor) | 0.1° / daily | 2015–2025 |

**Covariates (static):** Province-mean elevation, coastline fraction, forest cover fraction (from CNN land cover).
**Train / Val / Test split:** 2015–2023 training / last 90 days of 2023 validation / 2024 test.

---

## (c) Performance Metrics (2024 held-out test set, 7-day horizon)

### T2M (°C)
| Metric | Day 1 | Day 3 | Day 7 |
|---|---|---|---|
| RMSE (°C) | 0.71 | 0.98 | 1.18 |
| MAE (°C) | 0.49 | 0.72 | 0.89 |
| R² | 0.96 | 0.93 | 0.89 |

### RH (%)
| Metric | Day 1 | Day 3 | Day 7 |
|---|---|---|---|
| RMSE (%) | 3.2 | 5.1 | 7.4 |
| MAE (%) | 2.3 | 3.8 | 5.7 |
| R² | 0.91 | 0.87 | 0.79 |

**Promotion thresholds:** T2M RMSE ≤ 1.2°C AND RH RMSE ≤ 8% (Day 7) required for Production registry promotion.

---

## (d) Known Limitations & Failure Modes
- **ENSO transition months:** T2M skill degrades ~18% during rapid ENSO phase transitions when warm/cold pool positions shift anomalously; RH degradation up to 25% in maritime regions.
- **Urban heat island:** Province-averaged T2M masks urban micro-climate hotspots; DKI Jakarta province mean underestimates urban core by 1.2–1.8°C.
- **Extreme heat events:** Training contains limited >38°C events; tail-end heat stress events may be underestimated.
- **Mountain provinces:** Highland provinces (Papua Pegunungan, DI Yogyakarta highland) show higher RMSE due to station altitude mismatch with ERA5 grid.
- **Static covariate staleness:** Forest cover fraction covariate updated quarterly; rapid deforestation episodes introduce feature drift between retraining cycles.

---

## (e) Regional Fairness Analysis (Day 7 T2M RMSE °C)
| Region | RMSE °C | Notes |
|---|---|---|
| Sumatra | 1.09 | Good ERA5 coverage; coastal maritime variability |
| Jawa | 0.97 | Best; densest BMKG network; strongest training signal |
| Kalimantan | 1.14 | Equatorial low T2M variance; adequate performance |
| Sulawesi | 1.21 | Peninsula topography; 4-arm orographic complexity |
| Maluku + Papua | 1.38 | Highest; sparse BMKG stations; Papua highland altitude mismatch |

**Gap analysis:** Maluku+Papua shows 42% higher RMSE than Jawa. Heat-stress advisories for eastern Indonesia regions should be flagged with ±1.5°C uncertainty bounds in API-GATEWAY and VISUALIA outputs.

---

## (f) PP 71/2019 Data Residency Compliance
All training data, model artifacts, TFT weights, and forecast outputs processed and stored within Indonesian jurisdiction on **AWS ap-southeast-3 (Jakarta)**. No cross-border data transfer without BSSN authorization per PP 71/2019 Article 17. ERA5 data: ECMWF Copernicus open-access license (CDS). BMKG station data: MoU (2024-03-15). T2M and RH forecast outputs classified as *informasi meteorologi* under PP 71/2019 Article 7(1); public API distribution requires BMKG licensing acknowledgment. No PII processed. Heat-stress advisory outputs additionally subject to Kemenkes data sharing protocol (PKS Kemenkes-ANALYTICA 2025-01-10).

**Compliance status:** ✅ PP71_2019_COMPLIANT

---

## (g) Retraining Schedule
| Activity | Frequency |
|---|---|
| Drift check (PSI) | Daily — model_serving_health_check DAG |
| **Scheduled retraining** | **Quarterly** (triggered by model_serving_health_dag or manually via tft_training_job.py) |
| Emergency retrain | Automated on PSI CRITICAL |
| Performance review (T2M RMSE gate) | Each quarterly cycle — promotes to Production if RMSE ≤ 1.2°C else Staging |

MLflow experiment: `climate_transformer` | Registry alias: `tropi_climate_tft` | Training job: `src/training/tft_training_job.py`
