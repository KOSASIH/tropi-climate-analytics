# Model Card — XGBoost Precipitation Nowcast
**Model ID:** `xgboost_nowcast`
**Version:** See [MLflow Model Registry](http://mlflow:5000/#/models/xgboost_nowcast)
**Owner:** ANALYTICA | **Updated:** 2026-06-04

---

## Model Overview
Gradient-boosted tree ensemble (XGBoost) for 24–72 hour precipitation nowcasting over 38 Indonesian provinces. Ingests real-time multi-source climate features to deliver probabilistic rainfall forecasts at 4 km spatial resolution.

## Intended Use
| Use Case | Supported |
|---|---|
| 24–72 h precipitation nowcast (operational) | ✅ |
| Flood early warning input (HYDROLOGIS integration) | ✅ |
| Agricultural irrigation scheduling | ✅ |
| Climate projection beyond 72 h | ❌ |
| Areas outside Indonesian domain | ❌ |

**Intended users:** BMKG forecasters, HYDROLOGIS flood model inputs, national disaster agency (BNPB).

## Training Data
| Source | Variable(s) | Resolution | Period |
|---|---|---|---|
| GPM IMERG (NASA) | Precipitation (mm/h) | 0.1° / 30 min | 2015–2025 |
| MODIS MOD11A1 | Land surface temperature | 1 km / daily | 2015–2025 |
| SMAP L3 (NASA) | Soil moisture (m³/m³) | 36 km / daily | 2015–2025 |
| BMKG ground stations | T2M, RH, wind, pressure | Station / hourly | 2015–2025 |
| ECMWF ERA5 reanalysis | Upper-air profiles | 0.25° / hourly | 2015–2025 |

**Indonesian provinces covered:** All 38 provinces (Sumatra, Kalimantan, Java, Sulawesi, Maluku, Papua, Nusa Tenggara).
**Train/val/test split:** 2015–2022 / 2023 / 2024.

## Performance Metrics
| Metric | 24 h | 48 h | 72 h |
|---|---|---|---|
| RMSE (mm/h) | 0.84 | 1.12 | 1.43 |
| MAE (mm/h) | 0.51 | 0.71 | 0.94 |
| R² | 0.87 | 0.81 | 0.73 |
| CSI (threshold 5 mm/h) | 0.72 | 0.64 | 0.56 |
| FAR | 0.18 | 0.23 | 0.29 |

Metrics evaluated on 2024 held-out test set (n = 142,080 grid-hour samples).

## SHAP Feature Importances (Top 10)
| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | GPM rainfall t−1h | 0.312 |
| 2 | SMAP soil moisture | 0.187 |
| 3 | BMKG relative humidity | 0.143 |
| 4 | ERA5 850hPa specific humidity | 0.121 |
| 5 | Land surface temperature | 0.098 |
| 6 | BMKG surface pressure | 0.079 |
| 7 | ERA5 vertical velocity (omega) | 0.067 |
| 8 | BMKG wind speed | 0.054 |
| 9 | ENSO index (ONI) | 0.043 |
| 10 | IOD (Dipole Mode Index) | 0.038 |

SHAP values computed via TreeExplainer on 10,000 validation samples.

## Limitations & Risks
- **Orographic bias:** Under-predicts extreme rainfall on windward slopes (Bukit Barisan, Jayawijaya).
- **Convective initiation:** Limited skill for isolated deep convection onset (< 6 h lead time).
- **Data latency dependency:** Performance degrades when BMKG station feeds are delayed > 2 h.
- **ENSO extremes:** Training data includes only 3 El Niño / La Niña cycles; extreme ENSO conditions may extrapolate poorly.
- **Not calibrated for:** Snowfall (irrelevant in tropics), sub-1km urban flash floods.

## PP 71/2019 Compliance Notes
- All Indonesian ground observation data obtained under BMKG data-sharing MoU (signed 2024-03-15).
- NASA satellite data used under open-access license (EOSDIS).
- Model outputs classified as **meteorological information** under PP 71/2019 Article 7(2) — requires attribution to platform when disseminated publicly.
- Data residency: all training artifacts stored in AWS ap-southeast-3 (Jakarta) per PP 71/2019 data localization requirements.
- Personal data processing: model inputs contain no personally identifiable information (PII).

## Version History
| Version | MLflow Run | Date | Change |
|---|---|---|---|
| 1.0 | See MLflow registry | 2025-09-01 | Initial production release |
| 1.1 | See MLflow registry | 2025-12-15 | Added SMAP soil moisture feature; RMSE −8% |
| 2.0 | See MLflow registry | 2026-03-01 | Expanded to all 38 provinces; added IOD index |
| 2.1 | See MLflow registry | 2026-06-04 | Sprint 4 retrain; PSI drift check enforced |
