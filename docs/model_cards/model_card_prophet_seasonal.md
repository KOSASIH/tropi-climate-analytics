---
model_id: prophet_seasonal_climate
version: "2.0"
last_updated: "2026-06-04"
compliance_status: PP71_2019_COMPLIANT
contact: kosasihg88@gmail.com
---

# Model Card — Prophet Seasonal Climate Forecast
**Model ID:** `prophet_seasonal_climate` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## (a) Model Overview & Intended Use
Facebook Prophet time-series model with custom Indonesian-domain regressors (ENSO ONI, Indian Ocean Dipole, MJO phase index) for 30–180 day seasonal climate outlooks at province level. Captures additive trend, yearly/monthly seasonality, and ocean-atmosphere teleconnection signals.

| Use Case | Status |
|---|---|
| Monthly/seasonal rainfall outlook (30–180 d) | ✅ Supported |
| Agricultural seasonal planning (Kementan) | ✅ Supported |
| Drought probability index (SPI-3/SPI-6) | ✅ Supported |
| Sub-weekly temporal resolution | ❌ Not supported |
| Tropical cyclone track guidance | ❌ Not supported |

**End users:** Ministry of Agriculture (Kementan), BMKG seasonal desk, PU water resource managers, BNPB long-range disaster risk planning.

---

## (b) Training Data
**Spatial coverage:** All 38 Indonesian provinces (see xgb_precip card for full province list).

**Temporal range:** January 2000 – December 2025

| Source | Variable | Resolution | Period |
|---|---|---|---|
| GPM IMERG Monthly V07 (NASA) | Monthly accumulated precipitation mm | Province / monthly | 2000–2025 |
| NOAA CPC | ONI (ENSO 3.4 index) | Monthly | 2000–2025 |
| JAMSTEC | Indian Ocean Dipole Mode Index | Monthly | 2000–2025 |
| BMKG | Monthly T2M, RH climatologies | Province / monthly | 2000–2025 |
| ERA5 (ECMWF) | MJO RMM1/RMM2 phase index | Monthly | 2000–2025 |

**Train / Validation split:** 2000–2022 training / 2023–2024 validation (chronological, no leakage).

---

## (c) Performance Metrics (2024 held-out test set)
| Metric | 30-day | 90-day | 180-day |
|---|---|---|---|
| RMSE (mm/month) | 18.2 | 31.4 | 47.8 |
| MAE (mm/month) | 12.1 | 22.6 | 35.2 |
| R² | 0.83 | 0.71 | 0.58 |
| Drought onset F1 (SPI-3 ≤ −1.0) | 0.79 | 0.68 | 0.54 |

CSI not computed at monthly temporal resolution — binary precipitation CSI is meaningful only for sub-daily/daily scales.

---

## (d) Known Limitations & Failure Modes
- **ENSO transition months:** Skill degrades significantly during rapid ENSO onset/decay (Apr–Jun, Oct–Dec). RMSE increases ~28% during transition months as the model cannot anticipate non-linear ENSO phase shifts.
- **Regime shifts:** Non-stationary IOD behaviour post-2020 may reduce historical-pattern extrapolation reliability.
- **Province-only spatial scale:** Province-level forecasts only — not suitable for district-scale planning without statistical downscaling.
- **Anthropogenic warming trend:** Linear trend component may underestimate acceleration under SSP3/SSP5 trajectory; not a climate projection tool.
- **Land-use feedback:** Urban heat island and deforestation-driven rainfall feedbacks not explicitly parameterised.

---

## (e) Regional Fairness Analysis (90-day RMSE mm/month)
| Region | RMSE | Notes |
|---|---|---|
| Sumatra | 29.1 | Good ENSO teleconnection; peat swamp moisture feedback |
| Jawa | 26.8 | Best; well-defined two-season regime; dense validation data |
| Kalimantan | 34.2 | Weak seasonality signal → harder seasonal prediction |
| Sulawesi | 33.7 | Complex multi-regime rainfall (4-peak annual cycle) |
| Maluku + Papua | 41.5 | Bi-modal/tri-modal regime; sparse BMKG monthly climatology |

**Gap analysis:** Maluku+Papua 55% higher RMSE than Jawa. Root cause: station sparsity and complex bimodal/trimodal annual cycles. Caution advised for seasonal agricultural decisions in eastern Indonesia; recommend ensemble with ECMWF SEAS5 for these regions.

---

## (f) PP 71/2019 Data Residency Compliance
All data processed and stored within Indonesian jurisdiction on **AWS ap-southeast-3 (Jakarta)**. No cross-border data transfer without prior BSSN authorization per PP 71/2019 Article 17. BMKG monthly climatologies under bilateral MoU (PKS BMKG-ANALYTICA 2024-03-15). NOAA/JAMSTEC teleconnection indices are publicly released open data. Seasonal forecast outputs classified as *informasi iklim* under PP 71/2019 Article 7(3); distribution to government stakeholders requires BMKG co-branding per Article 12. No PII processed.

**Compliance status:** ✅ PP71_2019_COMPLIANT

---

## (g) Retraining Schedule
| Activity | Frequency |
|---|---|
| Drift check (PSI) | Daily — model_serving_health_check DAG |
| **Scheduled retraining** | **Monthly** (1st of each month, 03:00 WIB — incorporates new ENSO/IOD data) |
| Emergency retrain | Automated on PSI CRITICAL |
| ENSO regime review with BMKG | Quarterly |

MLflow experiment: `prophet_seasonal` | Registry alias: `tropi_prophet_seasonal` | DAG: `model_serving_health_check`
