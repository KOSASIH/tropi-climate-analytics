# ANALYTICA Model Documentation
**Tropi-Climate-Analytics | Sprint 0+1 | Version 0.1.0**
*Regulatory compliance documentation per PP Number 71/2019 (Indonesia)*

---

## 1. Overview

The ANALYTICA subsystem provides ML/AI predictive capabilities for the Tropi-Climate-Analytics platform. It comprises four production model families:

| Model | Algorithm | Horizon | Update Frequency |
|-------|-----------|---------|-----------------|
| Precipitation Nowcasting | XGBoost ensemble | 24/48/72h | Weekly retrain (Mon 01:00 WIB) |
| Seasonal Climate Forecasting | Prophet + ENSO regressors | 1–12 months | Monthly retrain (1st 02:00 WIB) |
| Satellite Classification | CNN (ResNet-lite) | Inference on demand | Quarterly retrain |
| Streamflow Forecasting | LSTM + Bahdanau attention | 6/12/24/48h | Monthly retrain |

---

## 2. Precipitation Nowcasting (XGBoost)

### 2.1 Purpose
24–72 hour precipitation accumulation forecasts at 4km resolution for flood early warning and BPBD alert integration.

### 2.2 Input Features
- **GPM IMERG** half-hourly estimates (bias-corrected via BMKG gauge fusion)
- **BMKG synoptic** observations: SLP, RH, wind speed/direction
- **SMAP** surface and root-zone soil moisture (6–48h lags)
- **DEM-derived** catchment morphology (slope, flow accumulation, aspect)
- **Temporal**: cyclical hour/DOY/month encodings, wet/dry season indicator

Feature count: ~420 per sample | Temporal lags: 1, 3, 6, 12, 24, 48, 72h | Rolling windows: 3, 6, 12, 24, 48h

### 2.3 Training Data
- Source: QPE fusion archive + BMKG historical gauges (Jan 2019 – present)
- Training window: 365 days rolling lookback
- Validation holdout: last 90 days (walk-forward)
- Spatial coverage: Indonesia (6°N–11°S, 95°E–141°E)

### 2.4 Performance Targets

| Metric | Threshold | Evaluation Period |
|--------|-----------|------------------|
| RMSE 24h | ≤ 8.0 mm | 90-day holdout |
| MAE 24h  | ≤ 5.0 mm | 90-day holdout |
| RMSE 48h | ≤ 12.0 mm | 90-day holdout |
| RMSE 72h | ≤ 15.0 mm | 90-day holdout |

### 2.5 MLflow Registry
- **Experiment**: `precipitation_nowcasting`
- **Registered model**: `precipitation-nowcasting-xgboost`
- **Stages**: Development → Staging → Production
- **Artifact storage**: `s3://tropi-climate-mlflow-artifacts/mlflow/precipitation_nowcasting/`

---

## 3. Seasonal Climate Forecasting (Prophet)

### 3.1 Purpose
1–12 month seasonal forecasts for temperature, monthly rainfall, SPI drought index, and agricultural water availability planning.

### 3.2 Model Configuration
- Yearly seasonality: enabled (Fourier order 10)
- Custom wet/dry seasonality: period 182.5 days (Fourier order 5)
- ENSO regressors: ONI index, IOD index, MJO phase (sin/cos)
- Changepoint prior scale: 0.05 (conservative for tropical signal)
- Seasonality mode: multiplicative

### 3.3 Variables Modelled

| Variable | Unit | Coverage |
|----------|------|----------|
| Monthly rainfall | mm | 500+ BMKG stations |
| Mean temperature | °C | 500+ BMKG stations |
| SPI-3 drought index | z-score | Gridded 10km |
| Agricultural water availability | mm/month | Provincial level |

### 3.4 Performance Target
- MAPE monthly rainfall: ≤ 15% on 90-day holdout

---

## 4. Satellite Classification (CNN)

### 4.1 Architecture
- 3 convolutional blocks (32 → 64 → 128 filters), 3×3 kernels, BatchNorm + ReLU
- AdaptiveAvgPool → FC(256) → Dropout(0.4) → output head
- Input: 64×64 pixel patches, 7 spectral bands (Landsat-8 OLI: B2–B7 + NDVI)

### 4.2 Tasks

| Task | Classes | Accuracy Target | Data Source |
|------|---------|----------------|-------------|
| Land cover classification | 10 (water, urban, crops, forest types, mangrove, bare soil...) | ≥ 85% | Landsat-8, 30m |
| Cloud masking | 2 (clear / cloud) | ≥ 92% | Landsat-8 + MODIS |
| Damage assessment | 3 (none / partial / severe) | ≥ 80% F1-macro | Sentinel-2 post-event |

### 4.3 Training Protocol
- Augmentation: horizontal/vertical flip, ±15° rotation, brightness ±20%
- Optimiser: AdamW (lr=1e-3, weight_decay=1e-4)
- LR schedule: CosineAnnealingLR (T_max=50 epochs)
- Class imbalance: weighted cross-entropy

---

## 5. Streamflow Forecasting (LSTM)

### 5.1 Architecture
- 3-layer stacked LSTM (hidden=256) + Bahdanau attention
- Lookback window: 30 days × 18 features
- Output heads: 6h, 12h, 24h, 48h streamflow (m³/s)

### 5.2 Station Coverage

| Station | River | Catchment Area | NSE Target |
|---------|-------|---------------|-----------|
| Manggarai | Ciliwung | 387 km² | ≥ 0.80 |
| Mlirip | Brantas | 11,800 km² | ≥ 0.80 |
| Jurug | Solo | 15,800 km² | ≥ 0.80 |

### 5.3 Features
GPM QPE (1/3/6/12/24h rolling), SMAP soil moisture, BMKG gauge (1/3/6/12/24h lags), DEM morphology (slope, FAC, TWI), calendar (hour/DOY cyclical, wet season flag)

---

## 6. MLOps Infrastructure

### 6.1 Retraining Schedule

| Model | Cron (UTC) | WIB Time | Trigger |
|-------|-----------|----------|---------|
| XGBoost precipitation | `0 18 * * 0` | Mon 01:00 | Weekly |
| Prophet seasonal | `0 19 1 * *` | 1st 02:00 | Monthly |
| CNN land cover | `0 20 1 1,4,7,10 *` | Quarterly 03:00 | Quarterly |
| LSTM streamflow | `0 19 1 * *` | 1st 02:00 | Monthly |

### 6.2 Quality Gate
Models must pass all performance thresholds before registration. Failed runs are logged to MLflow but not registered. Alert sent to `mlops-alerts@tropi-climate-analytics.id`.

### 6.3 A/B Testing
- Challenger receives 10% of inference traffic (deterministic SHA-256 hash routing)
- Promotion requires: ≥ 1,000 samples, Welch's t-test p < 0.05, challenger better on primary metric
- All routing decisions logged to MLflow for audit trail

### 6.4 Explainability (SHAP)
- TreeExplainer for XGBoost; KernelExplainer for others
- Top-5 SHAP drivers attached to each flood early-warning Kafka message
- Monthly compliance reports generated per PP Number 71/2019
- Reports stored at `s3://tropi-climate-mlflow-artifacts/compliance/`

---

## 7. Regulatory Compliance

| Requirement | Implementation |
|-------------|---------------|
| PP 71/2019 data governance | All training data sourced from BMKG/NASA with signed data agreements |
| Model auditability | Full MLflow run history; SHAP explanations per prediction |
| Data residency | All artifacts in AWS ap-southeast-3 (Jakarta) |
| Access control | RBAC via AWS IAM; model registry access logged |
| Retention | Training data 7 years; model artifacts 5 years |

---

*Last updated: 2026-06-04 | ANALYTICA v0.1.0 | Tropi-Climate-Analytics*
