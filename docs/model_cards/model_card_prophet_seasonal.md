---
model_id: prophet_seasonal_climate
version: "2.0"
last_updated: "2026-06-04"
compliance_status: compliant
contact: kosasihg88@gmail.com
---

# Model Card — Prophet Seasonal Climate Forecast
**Model ID:** `prophet_seasonal_climate` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## Model Overview & Intended Use
Facebook Prophet time-series model with custom Indonesian-domain regressors (ENSO, IOD, MJO) for 30–180 day seasonal climate outlooks at province level. Captures additive trend, yearly seasonality, and external ocean-atmosphere teleconnections.

| Use Case | Status |
|---|---|
| Monthly/seasonal rainfall outlook (30–180 d) | ✅ Supported |
| Agricultural seasonal planning | ✅ Supported |
| Drought probability index | ✅ Supported |
| Sub-weekly forecast | ❌ Not supported |
| Tropical cyclone tracks | ❌ Not supported |

**End users:** Ministry of Agriculture (Kementan), BMKG seasonal desk, water resource managers.

---

## Training Data
| Source | Variable | Resolution | Period |
|---|---|---|---|
| GPM IMERG | Monthly accumulated precipitation | Province / monthly | 2000–2025 |
| NOAA CPC | ONI (ENSO index) | Monthly | 2000–2025 |
| JAMSTEC | Indian Ocean Dipole Mode Index | Monthly | 2000–2025 |
| BMKG | Monthly T2M, RH climatologies | Province / monthly | 2000–2025 |
| ERA5 | MJO phase index | Monthly | 2000–2025 |

**Spatial coverage:** All 38 Indonesian provinces.  
**Train/val split:** 2000–2022 training / 2023–2024 validation.

---

## Performance Metrics
| Metric | 30-day | 90-day | 180-day |
|---|---|---|---|
| RMSE (mm/month) | 18.2 | 31.4 | 47.8 |
| MAE (mm/month) | 12.1 | 22.6 | 35.2 |
| R² | 0.83 | 0.71 | 0.58 |
| Drought onset F1 | 0.79 | 0.68 | 0.54 |

CSI not applicable (monthly temporal scale — CSI meaningful only for short-term binary precipitation).

---

## Known Limitations & Failure Modes
- **Regime shifts:** May miss rapid ENSO onset transitions mid-season.
- **Fine spatial scale:** Province-level only — not for district-scale planning without downscaling.
- **Warming trend:** Linear trend component may underestimate acceleration under high-emission scenarios.
- **Holiday/event effects:** Land-use change feedbacks not explicitly modeled.

---

## Fairness Analysis by Region (90-day RMSE mm/month)
| Region | RMSE | Notes |
|---|---|---|
| Sumatra | 29.1 | Good ENSO signal capture |
| Java | 26.8 | Best; two defined seasons |
| Kalimantan | 34.2 | Weak seasonality → harder to forecast |
| Sulawesi | 33.7 | Complex rainfall regime |
| Papua | 41.5 | Bi-modal rainfall, sparse data |

Papua and Kalimantan show the largest uncertainty due to complex orographic regimes and limited ground truth. Caution advised for seasonal agricultural decisions in these regions.

---

## PP 71/2019 Data Residency Compliance
- Data stored in **AWS ap-southeast-3 (Jakarta)**.
- BMKG data under bilateral MoU.
- Seasonal forecasts classified as **iklim information** under PP 71/2019 Article 7(3).
- Distribution to government stakeholders requires BMKG co-branding (PP 71/2019 Article 12).
- No PII processed.
- **Compliance status:** ✅ Compliant

---

## Monitoring & Retraining Schedule
| Activity | Frequency |
|---|---|
| PSI drift check | Daily — model_serving_health_check DAG |
| Model retraining | Monthly (new ENSO/IOD data) |
| Emergency retrain | Automated on PSI CRITICAL |
| ENSO regime review | Quarterly with BMKG seasonal forecasters |

MLflow experiment: `prophet_seasonal` | Registry: `prophet_seasonal_climate`
