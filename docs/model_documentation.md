# ANALYTICA Model Documentation
## Tropi-Climate-Analytics | Sprint 0

> **Agent**: ANALYTICA | **Sprint**: 0 | **Date**: 2026-06-04 | **Status**: Initial Release

---

## Overview

This document describes the ML/AI models developed by ANALYTICA for the Tropi-Climate-Analytics platform, covering architecture, data sources, performance targets, MLOps workflows, explainability, and regulatory compliance.

---

## 1. Model Inventory

| Model | Type | Horizon | Resolution | Target Metric |
|---|---|---|---|---|
| PrecipitationNowcastModel | XGBoost Regressor | 24–72 h | 0.25° | RMSE < 15 mm |
| SeasonalForecastModel | Prophet | 3–6 months | Province | MAPE < 20% |
| LandCoverCNN | ResNet + Attention CNN | N/A (classification) | 30 m | Accuracy ≥ 85% |

---

## 2. PrecipitationNowcastModel (XGBoost)

### Architecture
- **Algorithm**: XGBoost histogram gradient boosting (`tree_method=hist`)
- **Ensemble size**: 1,000 trees, max depth 8
- **Objective**: `reg:squarederror` with RMSE + MAE eval metrics
- **Regularisation**: L1 (α=0.1), L2 (λ=1.0), subsample 80%, colsample 80%
- **Early stopping**: 50 rounds on validation RMSE

### Feature Groups (8 groups, ~350 features)
| Group | Source | Features |
|---|---|---|
| MODIS cloud & land | MODIS MOD09/MOD11/MOD13 | cloud_optical_depth, ndvi, evi, LST, … |
| GPM precipitation | GPM IMERG | precip_rate_1/3/6h, latent_heat_flux, … |
| BMKG surface obs | BMKG AWS network | station_rainfall, T2m, RH, wind, MSLP, … |
| Atmospheric instability | NWP analysis | CAPE, CIN, LI, wind shear, PW, K-index, … |
| Topography | SRTM 30m | elevation, slope, aspect, TWI, dist_coast, … |
| Temporal (cyclic) | Derived | sin/cos hour, DOY, month, wet_season flag |
| Climate indices | NOAA/BOM | ONI, IOD, ENSO phase, MJO phase/amplitude |
| Precipitation lags | Self | lag 6/12/18/24/48/72 h + 6 rolling stats |

### Training Data
- **Period**: 2015–2024 (10 years)
- **Domain**: Indonesian archipelago (95°E–141°E, 11°S–6°N)
- **Positive/negative split**: Wet season 60%, dry season 40%
- **Train/val split**: Temporal — 2015–2022 train, 2023–2024 validation

### Performance (Sprint 0 target)
| Metric | Target | Evaluated on |
|---|---|---|
| RMSE | < 15 mm | 6-hour accumulation, 2023–2024 holdout |
| MAE | < 10 mm | same |
| Bias | ±2 mm | same |

---

## 3. SeasonalForecastModel (Prophet)

### Architecture
- **Algorithm**: Facebook Prophet with multiplicative seasonality
- **Seasonalities**: Yearly (built-in) + Indonesian semi-annual wet season (period=182.6 d, Fourier order 5) + ENSO-conditional annual cycle
- **Changepoints**: `changepoint_prior_scale=0.05` (conservative)
- **Regressors**: ONI, IOD, ENSO phase, MJO phase, MJO amplitude (all standardised)
- **Prediction intervals**: 95% credible interval

### Cross-Validation
- Initial training window: 730 days (2 years)
- Period between cutoffs: 180 days
- Forecast horizon: 90 days
- Metric: RMSE, MAE, MAPE averaged across all cutoffs

### Performance (Sprint 0 target)
| Metric | Target |
|---|---|
| CV RMSE | < 25 mm/month |
| CV MAPE | < 20% |

---

## 4. LandCoverCNN

### Architecture
```
Input: (64, 64, 7) — Landsat 8/9 OLI bands B1–B7 (scaled 0–1)
  → Data augmentation (RandomFlip, RandomRotation)
  → Conv2D(32) + BN
  → ResBlock(64) → MaxPool → Dropout(0.2)
  → ResBlock(128) → MaxPool → Dropout(0.2)
  → ResBlock(256) → MaxPool → Dropout(0.2)
  → Channel Attention (ratio=8)
  → ResBlock(512)
  → GlobalAveragePooling2D
  → Dense(256, ReLU) + Dropout(0.3)
  → Dense(9, Softmax)
Output: 9-class land cover probability vector
```

### Land Cover Classes
1. Forest  2. Degraded forest  3. Plantation  4. Cropland
5. Water  6. Urban  7. Bare land  8. Mangrove  9. Peatland

### Training Configuration
| Parameter | Value |
|---|---|
| Batch size | 64 |
| Max epochs | 100 (early stopping patience=15) |
| Optimiser | Adam (lr=1e-3, ReduceLROnPlateau patience=5) |
| Loss | Sparse categorical cross-entropy |
| L2 regularisation | 1e-4 |

### Performance (Sprint 0 target)
| Metric | Target |
|---|---|
| Overall accuracy | ≥ 85% |
| Per-class F1 (forest) | ≥ 0.90 |
| Per-class F1 (peatland) | ≥ 0.80 |

---

## 5. Feature Engineering Pipeline

The `FeatureEngineeringPipeline` class assembles **~1000+ features** per grid cell per timestep from 8 raw source groups:

- **Temporal cyclic encoding**: sin/cos for hour, DOY, month; wet-season binary flag
- **Precipitation lag features**: 9 lags (1h–168h)
- **Rolling statistics**: 6 windows × 3 stats (mean, max, std) = 18 features
- **Raw sensors**: 79 features from MODIS, GPM, BMKG, SMAP, atmospheric indices, topography, climate indices
- **Landsat spectral indices**: NDVI, EVI, NDWI, MNDWI, NBR, NDBI (from 7 OLI bands)
- **Interaction cross-terms**: CAPE×PW, NDVI×SM, shear×CAPE, ONI×IOD

Missing values are forward-filled then median-imputed. Optional z-score normalisation applied after `fit()`.

---

## 6. MLOps Workflow

### Tracking (MLflow)
- **Tracking URI**: `MLFLOW_TRACKING_URI` (default `http://localhost:5000`)
- **Experiment**: `tropi-climate-models`
- **Logged per run**: hyperparameters, train/val metrics, top-20 feature importances, model artefact, run tags

### Model Registry
| Registry Name | Model |
|---|---|
| `tropi-precipitation-nowcast` | PrecipitationNowcastModel |
| `tropi-seasonal-forecast` | SeasonalForecastModel |
| `tropi-land-cover-cnn` | LandCoverCNN |

Stages: `None` → `Staging` → `Production` → `Archived`

### Automated Retraining Schedule
| Model | Schedule | Timezone |
|---|---|---|
| XGBoost nowcast | Weekly — Monday 01:00 | Asia/Jakarta (WIB) |
| Prophet seasonal | Monthly — 1st of month 02:00 | Asia/Jakarta (WIB) |
| CNN land cover | Quarterly — 1st of month 03:00 every 3 months | Asia/Jakarta (WIB) |

### Retraining Trigger Logic
Retraining is triggered when **any** of the following is true:
1. **PSI > 0.2** on any feature between reference (training) and current distribution
2. **KS p-value < 0.05** on any feature
3. **Production RMSE increase > 10%** versus champion baseline

### Champion / Challenger A/B Protocol
1. New model trained → registered as `Staging` challenger
2. A/B evaluation: 5 rounds on held-out production data
3. If challenger win rate ≥ 60% → promoted to `Production`, champion archived
4. Otherwise → challenger archived, champion retained

---

## 7. Explainability

All models produce SHAP-based attributions via `ModelExplainer`:

| Model | SHAP Method | Output |
|---|---|---|
| XGBoost | TreeExplainer | Feature-level SHAP values, interaction effects |
| CNN | GradientExplainer | Per-band attribution maps (64×64 spatial) |
| Prophet | Component decomposition | Trend, seasonality, regressor contributions |

Top-10 feature attributions are logged to every MLflow run.
Full SHAP values available on-demand through the Analytics API (`/api/v1/models/{id}/explain`).

---

## 8. Regulatory Compliance

| Standard | Applicability |
|---|---|
| BMKG Technical Standard 2023 | Precipitation nowcasting outputs distributed to BMKG |
| KLHK PP-71/2019 | Land cover maps used by Ministry of Environment |
| NASA EOSDIS Data Use Policy | All NASA satellite data ingested under open-access terms |
| Google Model Card 2.0 | Model cards generated for each registered model version |

**Data lineage**: Tracked from raw satellite granule to prediction via MLflow run tags and input data SHA-256 hashes.
**Audit trail**: All training runs, data versions, and promotion decisions logged with immutable MLflow records.
**Retention**: Model artefacts retained 7 years per KLHK archiving policy.

---

## 9. Contacts & Governance

| Role | Contact |
|---|---|
| ML Engineering | ANALYTICA Agent |
| Data Engineering | DATA-FLOW Agent |
| Geospatial validation | GEOSPATIAL Agent |
| Atmospheric validation | ATMOSPHERE Agent |
| Executive oversight | CLIMATE-OS |

*Model promotion to Production requires CLIMATE-OS acknowledgement.*
