# Model Card — Prophet Seasonal Climate Forecast
**Model ID:** `prophet_seasonal`
**Version:** See [MLflow Model Registry](http://mlflow:5000/#/models/prophet_seasonal)
**Owner:** ANALYTICA | **Updated:** 2026-06-04

---

## Model Overview
Facebook Prophet time-series model with custom Indonesian-domain regressors for seasonal climate forecasting (30–180 day horizon). Captures additive trend, yearly/weekly seasonality, and external climate indices (ENSO, IOD, MJO) to produce province-level monthly outlooks.

## Intended Use
| Use Case | Supported |
|---|---|
| Monthly/seasonal rainfall outlook (30–180 d) | ✅ |
| Agricultural seasonal planning | ✅ |
| Drought probability index generation | ✅ |
| Sub-weekly forecast | ❌ |
| Tropical cyclone track prediction | ❌ |

**Intended users:** Ministry of Agriculture (Kementan), BMKG seasonal outlook desk, water resource managers.

## Training Data
| Source | Variable(s) | Resolution | Period |
|---|---|---|---|
| GPM IMERG (NASA) | Monthly accumulated precipitation | Province-level | 2000–2025 |
| NOAA CPC | ONI (ENSO index) | Monthly | 2000–2025 |
| JAMSTEC | Indian Ocean Dipole Mode Index | Monthly | 2000–2025 |
| BMKG | Monthly T2M, RH climatologies | Province-level | 2000–2025 |
| ERA5 | MJO phase index | Monthly | 2000–2025 |

**Indonesian provinces covered:** All 38 provinces.
**Train/val split:** 2000–2022 training / 2023–2024 validation.

## Performance Metrics
| Metric | 30-day | 90-day | 180-day |
|---|---|---|---|
| RMSE precipitation (mm/month) | 18.2 | 31.4 | 47.8 |
| MAE (mm/month) | 12.1 | 22.6 | 35.2 |
| R² | 0.83 | 0.71 | 0.58 |
| Drought onset detection (F1) | 0.79 | 0.68 | 0.54 |

Evaluated on 2023–2024 held-out period across all 38 provinces.

## SHAP Feature Importances (Top 10)
| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Yearly seasonality component | 0.421 |
| 2 | ONI (ENSO index) lag-1 month | 0.198 |
| 3 | IOD Dipole Mode Index | 0.156 |
| 4 | MJO phase (1–8) | 0.112 |
| 5 | T2M climatological anomaly | 0.089 |
| 6 | Trend component | 0.074 |
| 7 | SMAP 90-day soil moisture anomaly | 0.062 |
| 8 | Pacific Decadal Oscillation | 0.044 |
| 9 | Weekly seasonality | 0.031 |
| 10 | GPM lag-2 months | 0.028 |

## Limitations & Risks
- **Regime shifts:** May not detect sudden climatological regime shifts (e.g., rapid ENSO onset).
- **Fine spatial scale:** Province-level outputs only — not suitable for district-scale planning without downscaling.
- **Trend extrapolation:** Long-term trend component assumes linear warming; may underestimate acceleration under high-emission scenarios.
- **Holiday/event effects:** Anthropogenic land-use change not explicitly modeled.

## PP 71/2019 Compliance Notes
- Seasonal forecasts distributed to government stakeholders require BMKG co-branding under PP 71/2019 Article 12.
- All data sources carry open-access or bilateral MoU authorizations.
- Outputs classified as **iklim information** (climate information) — not legally binding weather warnings.
- Data residency: AWS ap-southeast-3 (Jakarta).

## Version History
| Version | MLflow Run | Date | Change |
|---|---|---|---|
| 1.0 | See MLflow registry | 2025-10-01 | Initial production |
| 1.1 | See MLflow registry | 2026-01-15 | Added MJO phase regressor; MAE −11% |
| 2.0 | See MLflow registry | 2026-06-04 | Sprint 4 retrain; automated drift guard |
