---
model_id: cnn_landcover_classifier
version: "1.3"
last_updated: "2026-06-04"
compliance_status: PP71_2019_COMPLIANT
contact: kosasihg88@gmail.com
---

# Model Card — CNN Land Cover Classifier
**Model ID:** `cnn_landcover_classifier` | **Owner:** ANALYTICA | **Contact:** kosasihg88@gmail.com

---

## (a) Model Overview & Intended Use
ResNet-50 CNN trained on multi-spectral Landsat 8/9 + MODIS composites to classify land cover across 10 KLHK-standard land use categories at 30 m resolution. Primary operational use: deforestation change detection, annual land cover mapping for KLHK SIMONTANA reporting, and GEOSPATIAL agent spatial analysis input.

| Use Case | Status |
|---|---|
| Land cover classification (10 classes, 30 m) | ✅ Supported |
| Deforestation / land change detection (annual Δ) | ✅ Supported |
| KLHK SIMONTANA annual baseline mapping | ✅ Supported |
| Sub-30 m classification | ❌ Not supported |
| Flood inundation extent mapping | ❌ Not supported |
| Night-time urban mapping | ❌ Not supported |

**End users:** KLHK (Ministry of Environment & Forestry), GEOSPATIAL agent, BNPB land-change risk monitoring, LAPAN earth observation program.

---

## (b) Training Data
**Spatial coverage:** All 38 Indonesian provinces.

**Temporal range:** January 2018 – December 2025

| Source | Band(s) | Resolution | Period |
|---|---|---|---|
| Landsat 8/9 OLI-TIRS (NASA/USGS) | B2–B7 (Blue, Green, Red, NIR, SWIR1, SWIR2) | 30 m / 16-day | 2018–2025 |
| MODIS MOD13Q1 V6.1 | NDVI, EVI 16-day composite | 250 m → 30 m bilinear | 2018–2025 |
| KLHK National Land Cover Map (Peta Penutupan Lahan) | Ground truth labels (10 classes) | 30 m / annual | 2018–2024 |
| BIG Topographic Map (RBI 1:25k) | Elevation, slope, aspect | 30 m (SRTM-derived) | Static |

**Training area:** All 38 provinces; training patches stratified by island group and land cover class.
**Total labeled patches:** ~2.1 M tiles (64×64 px, 30 m/px).
**Train / Val / Test:** 70% / 15% / 15% (stratified, no spatial leakage — train/test splits on different Landsat path/row).

---

## (c) Performance Metrics (2024 held-out test set)

### Overall
| Metric | Value |
|---|---|
| Overall Accuracy | 88.4% |
| Macro F1 | 0.864 |
| Weighted F1 | 0.883 |
| Cohen's Kappa | 0.871 |

### Per-class F1
| Class | F1 |
|---|---|
| Dense primary forest | 0.934 |
| Degraded/secondary forest | 0.847 |
| Plantation (monoculture) | 0.891 |
| Shrubland / bush | 0.832 |
| Grassland / savanna | 0.818 |
| Paddy cropland (sawah) | 0.903 |
| Dryland agriculture | 0.867 |
| Built-up / urban | 0.912 |
| Wetland / mangrove | 0.876 |
| Open land / bare soil | 0.858 |

**Deforestation alert CSI:** 0.821 (threshold: ≥20% dense-forest loss within 250 m tile, annual Δ).

---

## (d) Known Limitations & Failure Modes
- **Cloud contamination:** Persistent cloud cover in Papua (>65% cloud fraction annually) and Kalimantan peat fires reduce data availability; annual median compositing partially mitigates but cannot eliminate cloud gaps.
- **Seasonal phenology confusion:** Dryland / paddy confusion increases to F1=0.81 during off-peak growing season when spectral signatures converge.
- **Plantation vs. degraded forest:** East Kalimantan acacia/oil-palm plantations under heavy cloud produce spectral overlap with degraded forest; F1 drops to 0.83 in these sub-regions.
- **Rapid urbanization lag:** Built-up class may lag 6–12 months behind actual impervious area expansion in rapidly developing peri-urban zones.
- **Smoke interference:** Biomass burning smoke (Aug–Oct) introduces aerosol artefacts in optical bands; smoke pixels are masked using MODIS MOD14 fire mask but residual haze degrades classification quality.

---

## (e) Regional Fairness Analysis (Macro F1)
| Region | Macro F1 | Notes |
|---|---|---|
| Sumatra | 0.877 | Peat swamp / degraded forest confusion; fire smoke |
| Jawa | 0.912 | Best; low cloud, dense KLHK training labels |
| Kalimantan | 0.851 | Heavy smoke + cloud; plantation/forest confusion |
| Sulawesi | 0.858 | Complex terrain; adequate KLHK training coverage |
| Maluku + Papua | 0.821 | Lowest; persistent cloud + sparse KLHK ground truth; recommended SAR augmentation |

**Gap analysis:** Maluku+Papua 10% lower macro F1 than Jawa. Root cause: <30% clear-sky observation frequency. Roadmap item: fuse Sentinel-1 SAR (cloud-penetrating) for next version.

---

## (f) PP 71/2019 Data Residency Compliance
All satellite inputs, model weights, and inference outputs stored within Indonesian jurisdiction on **AWS ap-southeast-3 (Jakarta)**. No cross-border data transfer without BSSN authorization per PP 71/2019 Article 17. Landsat and MODIS: NASA EOSDIS open-access license. KLHK ground truth labels under PKS KLHK-ANALYTICA 2023. Model outputs classified as *informasi geospasial tematik* under PP 71/2019 Article 4; deforestation alerts must be cross-validated with KLHK SIMONTANA before official Government publication. No PII processed.

**Compliance status:** ✅ PP71_2019_COMPLIANT

---

## (g) Retraining Schedule
| Activity | Frequency |
|---|---|
| Drift check (PSI) | Daily — model_serving_health_check DAG |
| **Scheduled retraining** | **Quarterly** (post-KLHK annual label update cycle) |
| Emergency retrain | Automated on PSI CRITICAL |
| KLHK accuracy audit | Annually |

MLflow experiment: `cnn_land_cover` | Registry alias: `tropi_cnn_landcover` | DAG: `model_serving_health_check`
