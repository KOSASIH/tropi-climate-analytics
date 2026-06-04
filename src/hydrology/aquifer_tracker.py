"""
GRACE-FO Aquifer Depletion Tracker — Sprint 3 Deliverable 4
HYDROLOGIS | Tropi Climate Analytics

Monthly groundwater storage anomaly tracking for 5 critical aquifer systems:
  - North Jakarta (Aquifer Utara Jakarta)
  - Bandung Basin (CAT Bandung-Soreang)
  - Semarang (CAT Semarang-Demak)
  - Surabaya (CAT Surabaya-Bangkalan)
  - Makassar (CAT Makassar)

Data source: NASA GRACE-FO RL06 Mascon
  - workspace/data/grace/ (cached monthly NetCDF)
  - NASA CMR API pull (coordinate with DATA-FLOW for pipeline registration)

Method:
  TWSA (from GRACE-FO) → GWSA = TWSA − SMS_anomaly − SWE_anomaly
  SMS from GLDAS-NOAH v2.1; SWE = 0 for tropical Indonesia.
  Reference period: 2004-2009 (GRACE baseline).

Output:
  - AquiferReport Pydantic model per aquifer
  - JSON → workspace/output/aquifer/{aquifer_id}_{month}.json
  - Depletion alert: True when GWSA < -2.0 cm AND trend < -0.5 cm/yr

Run cadence: Monthly (Airflow DAG: grace_monthly).
# CRON: 0 6 3 * * (3rd of each month, 06:00 UTC)
# DAG wiring in Sprint 4 — document as comment above.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE  = os.getenv("WORKSPACE_ROOT", "workspace")
GRACE_DATA_DIR  = os.path.join(WORKSPACE, "data", "grace")
OUTPUT_DIR      = os.path.join(WORKSPACE, "output", "aquifer")
GLDAS_DATA_DIR  = os.path.join(WORKSPACE, "data", "gldas")

# Depletion alert thresholds
GWSA_ALERT_THRESHOLD_CM    = -2.0    # GWSA < -2.0 cm
TREND_ALERT_THRESHOLD_CMYR = -0.5   # trend < -0.5 cm/yr

# Reference period for anomaly baseline
REFERENCE_PERIOD = "2004-2009"
REFERENCE_PERIOD_YEARS = (2004, 2009)

# ---------------------------------------------------------------------------
# Aquifer definitions (BPS/ESDM Cekungan Air Tanah classification)
# ---------------------------------------------------------------------------

AQUIFER_REGISTRY: dict[str, dict] = {
    "north_jakarta": {
        "name": "Aquifer Utara Jakarta",
        "cat_id": "CAT-31-01",
        "province": "DKI Jakarta",
        "area_km2": 350.0,
        "population_served": 3_500_000,
        "depth_range_m": "10–120",   # confined aquifer depth
        "primary_use": "domestic/industrial",
        "centroid": (-6.12, 106.83),
        # GRACE-FO mascon cell nearest centroid
        "grace_lat": -6.0,
        "grace_lon": 107.0,
        # GLDAS soil depth limit (mm) for SMS extraction
        "gldas_soil_depth_mm": 2000.0,
        # Baseline (2004-2009) GWSA stats derived from GRACE observations
        "baseline_gwsa_mean_cm": 0.0,
        "baseline_gwsa_std_cm":  2.8,
        # Known depletion rate from ESDM monitoring (cm/yr)
        "known_trend_cm_per_year": -3.2,
        "regulatory_status": "Kritis (Kepgub DKI 115/2020)",
    },
    "bandung_basin": {
        "name": "CAT Bandung-Soreang",
        "cat_id": "CAT-32-03",
        "province": "Jawa Barat",
        "area_km2": 1750.0,
        "population_served": 8_200_000,
        "depth_range_m": "30–250",
        "primary_use": "domestic/textile industry",
        "centroid": (-6.92, 107.61),
        "grace_lat": -7.0,
        "grace_lon": 108.0,
        "gldas_soil_depth_mm": 2000.0,
        "baseline_gwsa_mean_cm": 0.0,
        "baseline_gwsa_std_cm":  3.5,
        "known_trend_cm_per_year": -1.8,
        "regulatory_status": "Rawan (ESDM Monitoring 2023)",
    },
    "semarang": {
        "name": "CAT Semarang-Demak",
        "cat_id": "CAT-33-04",
        "province": "Jawa Tengah",
        "area_km2": 410.0,
        "population_served": 2_100_000,
        "depth_range_m": "20–180",
        "primary_use": "domestic/PDAM",
        "centroid": (-7.00, 110.42),
        "grace_lat": -7.0,
        "grace_lon": 110.0,
        "gldas_soil_depth_mm": 2000.0,
        "baseline_gwsa_mean_cm": 0.0,
        "baseline_gwsa_std_cm":  2.5,
        "known_trend_cm_per_year": -2.5,
        "regulatory_status": "Kritis — land subsidence 6.5 cm/yr (BRIN 2023)",
    },
    "surabaya": {
        "name": "CAT Surabaya-Bangkalan",
        "cat_id": "CAT-35-02",
        "province": "Jawa Timur",
        "area_km2": 820.0,
        "population_served": 4_600_000,
        "depth_range_m": "15–200",
        "primary_use": "domestic/industry/port",
        "centroid": (-7.25, 112.75),
        "grace_lat": -7.0,
        "grace_lon": 113.0,
        "gldas_soil_depth_mm": 2000.0,
        "baseline_gwsa_mean_cm": 0.0,
        "baseline_gwsa_std_cm":  3.0,
        "known_trend_cm_per_year": -1.5,
        "regulatory_status": "Rawan (PDAM Surya monitoring)",
    },
    "makassar": {
        "name": "CAT Makassar",
        "cat_id": "CAT-73-01",
        "province": "Sulawesi Selatan",
        "area_km2": 560.0,
        "population_served": 1_800_000,
        "depth_range_m": "20–150",
        "primary_use": "domestic/PDAM Makassar",
        "centroid": (-5.14, 119.43),
        "grace_lat": -5.0,
        "grace_lon": 119.0,
        "gldas_soil_depth_mm": 2000.0,
        "baseline_gwsa_mean_cm": 0.0,
        "baseline_gwsa_std_cm":  2.2,
        "known_trend_cm_per_year": -1.0,
        "regulatory_status": "Normal (ESDM 2022)",
    },
}

# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------

class AquiferReport(BaseModel):
    aquifer_id:          str
    aquifer_name:        str
    cat_id:              str
    province:            str
    area_km2:            float
    population_served:   int
    gwsa_cm:             float     = Field(..., description="Groundwater storage anomaly (cm eq. water)")
    twsa_cm:             float     = Field(..., description="Total water storage anomaly from GRACE-FO (cm)")
    sms_anomaly_cm:      float     = Field(..., description="Soil moisture storage anomaly subtracted (cm)")
    trend_cm_per_year:   float     = Field(..., description="Linear GWSA trend (cm/yr; negative = depletion)")
    gwsa_zscore:         float     = Field(..., description="GWSA normalised by 2004-2009 std dev")
    depletion_alert:     bool      = Field(..., description="True when GWSA < -2cm AND trend < -0.5 cm/yr")
    alert_severity:      str       = Field(..., description="NORMAL / WATCH / WARNING / CRITICAL")
    reference_period:    str       = "2004-2009"
    grace_solution:      str       = "CSR RL06M Mascon"
    data_month:          date
    regulatory_status:   str
    report_generated_at: datetime


class AquiferTrackerRunStatus(BaseModel):
    run_time_utc:        datetime
    data_month:          date
    aquifers_processed:  int
    aquifers_alert:      list[str]
    aquifers_critical:   list[str]
    output_paths:        list[str]
    success:             bool
    error:               Optional[str] = None


# ---------------------------------------------------------------------------
# GRACE-FO data loader (production path)
# ---------------------------------------------------------------------------

def _load_grace_twsa(
    lat: float,
    lon: float,
    data_month: date,
    grace_dir: str = GRACE_DATA_DIR,
) -> float:
    """
    Load GRACE-FO TWSA (cm) for nearest mascon cell at (lat, lon) for data_month.

    Production:
      1. Check grace_dir for cached CSR RL06M mascon NetCDF:
         {grace_dir}/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_YYYY-MM.nc
      2. If not cached, pull from NASA CMR API (coordinate with DATA-FLOW):
         https://cmr.earthdata.nasa.gov/search/granules → ShortName=TELLUS_GRAC-GRFO_MASCON_CRI_TIME_SERIES_RL06_V3
      3. Extract TWSA value at nearest 0.5° mascon cell.

    Sprint 3 stub: synthetic TWSA from known trend + noise.
    """
    meta = next(
        (v for v in AQUIFER_REGISTRY.values()
         if abs(v["grace_lat"] - lat) < 0.5 and abs(v["grace_lon"] - lon) < 0.5),
        None
    )
    if meta is None:
        return 0.0

    trend = meta["known_trend_cm_per_year"]
    years_since_base = (data_month.year - 2015) + data_month.month / 12.0
    seasonal = 1.2 * np.sin(2 * np.pi * (data_month.month - 3) / 12)
    rng = np.random.default_rng(int(lat * 100 + lon * 100) + data_month.toordinal())
    noise = float(rng.normal(0, 1.5))
    return round(trend * years_since_base + seasonal + noise, 2)


def _load_gldas_sms_anomaly(
    lat: float,
    lon: float,
    data_month: date,
) -> float:
    """
    Load GLDAS-NOAH v2.1 soil moisture storage anomaly (cm) relative to 2004-2009 mean.

    Production: read from workspace/data/gldas/ NetCDF or DATA-FLOW feature store.
    Sprint 3 stub: seasonal SMS from climatological cycle.
    """
    seasonal = 0.8 * np.sin(2 * np.pi * (data_month.month - 4) / 12)
    rng = np.random.default_rng(int(lat * 50 + lon * 50) + data_month.toordinal() + 999)
    return round(seasonal + float(rng.normal(0, 0.5)), 2)


# ---------------------------------------------------------------------------
# Depletion alert classifier
# ---------------------------------------------------------------------------

def _classify_alert_severity(gwsa_cm: float, trend: float, zscore: float) -> str:
    if gwsa_cm < -5.0 and trend < -2.0:
        return "CRITICAL"
    if gwsa_cm < -3.0 or (gwsa_cm < -2.0 and trend < -1.0):
        return "WARNING"
    if gwsa_cm < -1.0 or trend < -0.5:
        return "WATCH"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class GRACEFOAquiferTracker:
    """
    Monthly GRACE-FO aquifer depletion tracker for 5 Indonesian critical aquifer systems.

    Usage (from Airflow grace_monthly DAG):
        tracker = GRACEFOAquiferTracker()
        status = tracker.run(data_month=date(2026, 6, 1))

    # CRON schedule: 0 6 3 * * Asia/Jakarta — run on 3rd of each month
    # DAG wiring: Sprint 4, file: dags/grace_monthly.py
    """

    def __init__(self) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(GRACE_DATA_DIR, exist_ok=True)
        os.makedirs(GLDAS_DATA_DIR, exist_ok=True)

    def run(self, data_month: date) -> AquiferTrackerRunStatus:
        run_time = datetime.now(timezone.utc)
        reports:      list[AquiferReport] = []
        output_paths: list[str] = []
        alert_list:    list[str] = []
        critical_list: list[str] = []

        for aquifer_id, meta in AQUIFER_REGISTRY.items():
            lat = meta["grace_lat"]
            lon = meta["grace_lon"]

            # 1. Pull TWSA from GRACE-FO
            twsa_cm = _load_grace_twsa(lat, lon, data_month)

            # 2. Pull GLDAS SMS anomaly
            sms_cm  = _load_gldas_sms_anomaly(lat, lon, data_month)

            # SWE = 0 (tropical Indonesia)
            gwsa_cm = round(twsa_cm - sms_cm, 2)

            # 3. Compute trend (linear fit over rolling 24-month window)
            # Sprint 3 stub: use known trend from ESDM data
            trend = meta["known_trend_cm_per_year"]

            # 4. Normalise by 2004-2009 std dev
            std = meta["baseline_gwsa_std_cm"]
            zscore = round(gwsa_cm / std, 3) if std > 0 else 0.0

            # 5. Alert
            severity   = _classify_alert_severity(gwsa_cm, trend, zscore)
            dep_alert  = gwsa_cm < GWSA_ALERT_THRESHOLD_CM and trend < TREND_ALERT_THRESHOLD_CMYR

            report = AquiferReport(
                aquifer_id=aquifer_id,
                aquifer_name=meta["name"],
                cat_id=meta["cat_id"],
                province=meta["province"],
                area_km2=meta["area_km2"],
                population_served=meta["population_served"],
                gwsa_cm=gwsa_cm,
                twsa_cm=twsa_cm,
                sms_anomaly_cm=sms_cm,
                trend_cm_per_year=trend,
                gwsa_zscore=zscore,
                depletion_alert=dep_alert,
                alert_severity=severity,
                reference_period=REFERENCE_PERIOD,
                data_month=data_month,
                regulatory_status=meta["regulatory_status"],
                report_generated_at=run_time,
            )
            reports.append(report)

            path = self._write_output(aquifer_id, data_month, report)
            output_paths.append(path)

            if severity in ("WARNING", "CRITICAL"):
                alert_list.append(meta["name"])
            if severity == "CRITICAL":
                critical_list.append(meta["name"])

        logger.info(
            "Aquifer tracker complete | month=%s alert=%d critical=%d",
            data_month, len(alert_list), len(critical_list),
        )

        return AquiferTrackerRunStatus(
            run_time_utc=run_time,
            data_month=data_month,
            aquifers_processed=len(AQUIFER_REGISTRY),
            aquifers_alert=alert_list,
            aquifers_critical=critical_list,
            output_paths=output_paths,
            success=True,
        )

    def _write_output(
        self,
        aquifer_id: str,
        data_month: date,
        report: AquiferReport,
    ) -> str:
        fname = f"{aquifer_id}_{data_month.strftime('%Y%m')}.json"
        path  = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as fh:
            json.dump(report.model_dump(), fh, indent=2, default=str)
        logger.info("Aquifer report written: %s", path)
        return path
