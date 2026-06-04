---
model_id: xgb_precip_nowcast
version: "2.1"
last_updated: "2026-06-04"
compliance_status: PP71_2019_COMPLIANT
contact: kosasihg88@gmail.com
---

# Model Card — XGBoost Precipitation Nowcast
**Model ID:** `xgb_precip_nowcast` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## (a) Model Overview & Intended Use
Gradient-boosted tree ensemble (XGBoost 2.x) for 24–72 hour precipitation nowcasting across all 38 Indonesian provinces. Ingests real-time multi-source climate features and delivers probabilistic rainfall forecasts at 4 km spatial resolution.

| Use Case | Status |
|---|---|
| 24–72 h precipitation nowcast (operational) | ✅ Supported |
| Flood early warning input (HYDROLOGIS) | ✅ Supported |
| Agricultural irrigation scheduling | ✅ Supported |
| Projection beyond 72 h | ❌ Not supported |
| Regions outside Indonesia | ❌ Not supported |

**End users:** BMKG forecasters, HYDROLOGIS flood alert pipeline, BNPB disaster preparedness, Kementan agricultural planning.

---

## (b) Training Data
**Spatial coverage:** All 38 Indonesian provinces — Aceh, Sumatera Utara, Sumatera Barat, Riau, Kepulauan Riau, Jambi, Bengkulu, Sumatera Selatan, Kepulauan Bangka Belitung, Lampung, Banten, DKI Jakarta, Jawa Barat, Jawa Tengah, DI Yogyakarta, Jawa Timur, Bali, Nusa Tenggara Barat, Nusa Tenggara Timur, Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan, Kalimantan Timur, Kalimantan Utara, Sulawesi Utara, Gorontalo, Sulawesi Tengah, Sulawesi Barat, Sulawesi Selatan, Sulawesi Tenggara, Maluku Utara, Maluku, Papua Barat Daya, Papua Barat, Papua Tengah, Papua Pegunungan, Papua Selatan, Papua.

**Temporal range:** January 2015 – December 2025

| Source | Variable | Resolution | Period |
|---|---|---|---|
| GPM IMERG Final V07 (NASA) | Precipitation mm/h | 0.1° / 30 min | 2015–2025 |
| MODIS MOD11A1 V6.1 | Land surface temperature K | 1 km / daily | 2015–2025 |
| SMAP L3 Enhanced (NASA) | Soil moisture m³/m³ | 9 km / daily | 2015–2025 |
| BMKG ground stations (210 stations) | T2M, RH, wind speed/dir, MSLP | Station / hourly | 2015–2025 |
| ERA5 reanalysis (ECMWF) | 500/850 hPa winds, Q, GPH | 0.25° / hourly | 2015–2025 |

**Train / Validation / Test split:** 2015–2022 / 2023 / 2024 (chronological, no leakage).

---

## (c) Performance Metrics (2024 held-out test set)
| Metric | 24 h | 48 h | 72 h |
|---|---|---|---|
| RMSE (mm/h) | 0.84 | 1.12 | 1.43 |
| MAE (mm/h) | 0.51 | 0.71 | 0.94 |
| R² | 0.87 | 0.81 | 0.73 |
| CSI (≥5 mm/h threshold) | 0.72 | 0.64 | 0.56 |
| FAR | 0.18 | 0.23 | 0.29 |

---

## (d) Known Limitations & Failure Modes
- **Orographic bias:** Under-predicts extreme rainfall on windward slopes (Bukit Barisan, Jayawijaya range). Bias +18% on slopes > 15°.
- **Deep convective initiation:** Skill limited for isolated convective onset with < 6 h lead. CSI drops to 0.41 for convection-only events.
- **ENSO transition months:** Forecast skill degrades during ENSO onset (Apr–Jun, Oct–Dec) when anomalous moisture transport disrupts climatological patterns. RMSE increases ~22% during transition months.
- **Data latency:** Accuracy degrades measurably when BMKG station feeds are delayed > 2 h; fallback to ERA5 adds ~0.15 mm/h RMSE.
- **Papua remote basins:** Limited ground truth in inland Papua; GPM IMERG F-CAL adjustment less reliable in steep terrain.

---

## (e) Regional Fairness Analysis (24 h RMSE mm/h)
| Region | RMSE | Notes |
|---|---|---|
| Sumatra | 0.91 | Orographic convection over Bukit Barisan reduces skill |
| Jawa | 0.78 | Best performance; densest BMKG station network |
| Kalimantan | 0.88 | Equatorial diurnal cycle variability; low orographic bias |
| Sulawesi | 0.96 | Complex peninsula terrain degrades skill |
| Maluku + Papua | 1.14 | Sparse BMKG coverage; Jayawijaya orographic bias |

**Gap analysis:** Maluku+Papua shows 46% higher RMSE than Jawa. Mitigation: augmenting with additional BMKG telemetry stations and Sentinel-1 SAR precipitation proxy is recommended before full operational alert deployment in these regions.

---

## (f) PP 71/2019 Data Residency Compliance
All training data, model artifacts, inference outputs, and feature store data are stored exclusively within Indonesian jurisdiction on **AWS ap-southeast-3 (Jakarta)**. No cross-border data transfer occurs without prior written authorization from BSSN (Badan Siber dan Sandi Negara) per PP 71/2019 Article 17. NASA satellite data accessed via EOSDIS open-access license. BMKG station data processed under bilateral MoU (signed 2024-03-15, valid through 2027). Outputs classified as meteorological information per PP 71/2019 Article 7(2) — public dissemination requires BMKG attribution. No personally identifiable information (PII) is processed at any stage.

**Compliance status:** ✅ PP71_2019_COMPLIANT

---

## (g) Retraining Schedule
| Activity | Frequency |
|---|---|
| Drift check (PSI all features) | Daily — model_serving_health_check DAG 06:00 WIB |
| **Scheduled retraining** | **Weekly** (every Monday 02:00 WIB) |
| Emergency retrain | Automated when PSI > 0.25 (CRITICAL threshold) |
| Performance review with BMKG | Quarterly |

MLflow experiment: `xgboost_nowcast` | Registry alias: `tropi_xgb_precip` | DAG: `model_serving_health_check`
