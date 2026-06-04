---
model_id: xgb_precip_nowcast
version: "2.1"
last_updated: "2026-06-04"
compliance_status: compliant
contact: kosasihg88@gmail.com
---

# Model Card — XGBoost Precipitation Nowcast
**Model ID:** `xgb_precip_nowcast` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## Model Overview & Intended Use
Gradient-boosted tree ensemble (XGBoost) for 24–72 hour precipitation nowcasting across 38 Indonesian provinces. Ingests real-time multi-source climate features and delivers probabilistic rainfall forecasts at 4 km spatial resolution.

| Use Case | Status |
|---|---|
| 24–72 h precipitation nowcast (operational) | ✅ Supported |
| Flood early warning input (HYDROLOGIS) | ✅ Supported |
| Agricultural irrigation scheduling | ✅ Supported |
| Projection beyond 72 h | ❌ Not supported |
| Regions outside Indonesia | ❌ Not supported |

**End users:** BMKG forecasters, HYDROLOGIS flood inputs, BNPB disaster agency.

---

## Training Data
| Source | Variable | Resolution | Period |
|---|---|---|---|
| GPM IMERG (NASA) | Precipitation mm/h | 0.1° / 30 min | 2015–2025 |
| MODIS MOD11A1 | Land surface temperature | 1 km / daily | 2015–2025 |
| SMAP L3 (NASA) | Soil moisture m³/m³ | 36 km / daily | 2015–2025 |
| BMKG ground stations | T2M, RH, wind, pressure | Station / hourly | 2015–2025 |
| ERA5 reanalysis | Upper-air profiles | 0.25° / hourly | 2015–2025 |

**Spatial coverage:** All 38 Indonesian provinces.  
**Train/val/test split:** 2015–2022 / 2023 / 2024.

---

## Performance Metrics (2024 held-out test set)
| Metric | 24 h | 48 h | 72 h |
|---|---|---|---|
| RMSE (mm/h) | 0.84 | 1.12 | 1.43 |
| MAE (mm/h) | 0.51 | 0.71 | 0.94 |
| R² | 0.87 | 0.81 | 0.73 |
| CSI (5 mm/h threshold) | 0.72 | 0.64 | 0.56 |
| FAR | 0.18 | 0.23 | 0.29 |

---

## Known Limitations & Failure Modes
- **Orographic bias:** Under-predicts extreme rainfall on windward slopes (Bukit Barisan, Jayawijaya).
- **Deep convection onset:** Skill limited for isolated convective initiation < 6 h lead time.
- **Data latency:** Accuracy degrades when BMKG feeds are delayed > 2 h.
- **Extreme ENSO:** Training includes only 3 El Niño/La Niña cycles — extreme events may extrapolate poorly.
- **Not calibrated for:** Urban flash floods below 1 km resolution.

---

## Fairness Analysis by Region (24 h RMSE mm/h)
| Region | RMSE | Notes |
|---|---|---|
| Sumatra | 0.91 | Higher due to orographic convection |
| Java | 0.78 | Best performance; densest BMKG network |
| Kalimantan | 0.88 | Equatorial diurnal cycle variability |
| Sulawesi | 0.96 | Complex terrain degrades skill |
| Papua | 1.14 | Sparse ground observations; worst coverage |

Papua shows the largest performance gap due to sparse BMKG telemetry. Recommended: augment with BMKG additional stations before operational alert use in Papua.

---

## PP 71/2019 Data Residency Compliance
- All training artifacts and model outputs stored in **AWS ap-southeast-3 (Jakarta)** per PP 71/2019 data localization requirements.
- NASA satellite data used under open-access EOSDIS license.
- BMKG data used under MoU signed 2024-03-15.
- Outputs classified as **meteorological information** under PP 71/2019 Article 7(2) — require attribution when disseminated publicly.
- No personally identifiable information (PII) processed.
- **Compliance status:** ✅ Compliant

---

## Monitoring & Retraining Schedule
| Activity | Frequency |
|---|---|
| PSI drift check (all features) | Daily — model_serving_health_check DAG 06:00 WIB |
| Model retraining (scheduled) | Monthly |
| Emergency retrain (PSI > 0.25 CRITICAL) | Automated — Airflow retraining DAG trigger |
| Performance review | Quarterly |

MLflow experiment: `xgboost_nowcast` | Registry: `xgb_precip_nowcast` | Monitoring DAG: `model_serving_health_check`
