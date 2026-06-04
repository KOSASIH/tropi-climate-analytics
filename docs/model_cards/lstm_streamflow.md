# Model Card — LSTM Streamflow Forecast
**Model ID:** `lstm_streamflow`
**Version:** See [MLflow Model Registry](http://mlflow:5000/#/models/lstm_streamflow)
**Owner:** ANALYTICA | **Updated:** 2026-06-04

---

## Model Overview
Stacked bidirectional LSTM for 6–72 hour river streamflow and flood stage forecasting across major Indonesian watersheds. Integrates HYDROLOGIS QPE (quantitative precipitation estimates), SMAP soil moisture, and real-time BMKG gauge telemetry to predict discharge (m³/s) and flood stage (m) at 23 river stations across 12 watersheds.

## Intended Use
| Use Case | Supported |
|---|---|
| 6–72 h streamflow/flood stage forecast | ✅ |
| Flood early warning (HYDROLOGIS integration) | ✅ |
| Seasonal low-flow / drought risk prediction | ✅ |
| Dam safety inflow forecasting | ✅ |
| Sub-6 h nowcasting | ⚠️ (experimental) |
| Tidal/coastal flooding | ❌ |
| Watershed outside training domain | ❌ |

**Intended users:** HYDROLOGIS agent, BNPB (disaster agency), PU (Ministry of Public Works / BBWS), BMKG flood desk.

## Training Data
| Source | Variable(s) | Resolution | Period |
|---|---|---|---|
| HYDROLOGIS QPE (GPM IMERG + BMKG blend) | Rainfall upstream (mm/h) | 4 km / 30 min | 2015–2025 |
| SMAP L3 Enhanced | Soil moisture (m³/m³) | 9 km / 3-day | 2015–2025 |
| BMKG Telemetry (AWLR) | River stage (m), discharge (m³/s) | Station / 15 min | 2015–2025 |
| DEM SRTM 30 m | Upstream catchment area, slope, TWI | Static | — |
| GEOSPATIAL land cover | Imperviousness fraction | 30 m / annual | 2015–2025 |

**Watersheds covered (12):** Ciliwung-Cisadane (Jakarta), Brantas (East Java), Solo (Central Java), Citarum (West Java), Mahakam (Kalimantan), Barito, Kapuas, Musi (Sumatra), Kampar, Batanghari, Rokan, Membramo (Papua).
**River stations:** 23 BMKG AWLR telemetry stations.
**Train/val/test split:** 2015–2022 / 2023 / 2024. Rolling 72-hour sequences, stride=1 h.

## Performance Metrics

### Streamflow (discharge m³/s) — 2024 test set
| Lead Time | RMSE (m³/s) | NSE | KGE | PBIAS |
|---|---|---|---|---|
| 6 h | 84 | 0.91 | 0.89 | −3.1% |
| 24 h | 142 | 0.86 | 0.83 | −5.4% |
| 48 h | 213 | 0.78 | 0.75 | −8.2% |
| 72 h | 291 | 0.71 | 0.68 | −11.7% |

**NSE** = Nash–Sutcliffe Efficiency | **KGE** = Kling–Gupta Efficiency | **PBIAS** = percent bias.

### Flood Stage (m)
| Lead Time | RMSE (m) | R² |
|---|---|---|
| 6 h | 0.18 | 0.94 |
| 24 h | 0.34 | 0.88 |
| 48 h | 0.52 | 0.81 |
| 72 h | 0.73 | 0.72 |

### Flood Event Detection (threshold: alert stage)
| Metric | Value |
|---|---|
| Probability of Detection (POD) | 0.884 |
| False Alarm Rate (FAR) | 0.107 |
| Critical Success Index (CSI) | 0.806 |
| Lead time (first alert before peak) | 14.2 h median |

## SHAP Feature Importances (Top 10)
| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Upstream rainfall t−3h (QPE) | 0.341 |
| 2 | River stage t−1h (autoregressive) | 0.287 |
| 3 | SMAP soil moisture (antecedent) | 0.198 |
| 4 | Upstream rainfall t−6h | 0.162 |
| 5 | Catchment imperviousness fraction | 0.121 |
| 6 | QPE 24-hour accumulation | 0.112 |
| 7 | Upstream stage (tributary) | 0.098 |
| 8 | DEM-derived topographic wetness index | 0.087 |
| 9 | BMKG RH (moisture availability) | 0.063 |
| 10 | Day-of-year (seasonal encoding) | 0.044 |

## Limitations & Risks
- **Data gaps:** AWLR station outages > 3 h degrade forecast quality significantly; fallback uses climatological prior.
- **Extreme flood extrapolation:** Events exceeding 2015 Ciliwung flood magnitude are outside training distribution.
- **Urbanization lag:** Annual land cover updates may lag actual imperviousness changes in rapidly urbanizing areas.
- **QPE error propagation:** GPM-IMERG underestimates localized convective cells; HYDROLOGIS QPE blend reduces but does not eliminate this.
- **Not calibrated for:** Tidal backwater effects (lower Ciliwung tidal influence), dam gate operations (requires BBWS real-time gate data feed).
- **HYDROLOGIS dependency:** Requires HYDROLOGIS QPE feed < 1-hour latency for operational use.

## PP 71/2019 Compliance Notes
- BMKG AWLR telemetry data used under BMKG Data-Sharing MoU (renewed 2024-09-01).
- Flood forecast outputs classified as **early warning information** under PP 71/2019 Article 8; dissemination to public requires BMKG operational validation before release.
- Model does not store individual citizen data; all inputs are hydrometeorological sensor readings (no PII).
- Data residency: AWS ap-southeast-3 (Jakarta).
- Flood alert generation pipeline audited by BNPB technical team (audit ref: BNPB-2025-AUDIT-042).

## Version History
| Version | MLflow Run | Date | Change |
|---|---|---|---|
| 1.0 | See MLflow registry | 2025-09-15 | Initial production — Ciliwung & Brantas only |
| 1.1 | See MLflow registry | 2026-01-20 | Expanded to 12 watersheds, 23 stations; NSE +0.07 |
| 1.2 | See MLflow registry | 2026-03-10 | SMAP soil moisture integration; FAR −0.03 |
| 2.0 | See MLflow registry | 2026-06-04 | Sprint 4 retrain; HYDROLOGIS weight-delivery wiring; drift guard |
