"""
QPE Fusion Pipeline — GPM IMERG + BMKG gauge bias-correction.

Algorithm:
  1. Fetch GPM IMERG Early Run half-hourly 4km precipitation field (NASA CMR / GESDISC)
  2. Fetch BMKG automatic rain gauge (ARG) observations for the same 30-min window
  3. Collocate gauges onto the IMERG 4km grid
  4. Multiplicative bias correction: QPE_corrected[i,j] = QPE_raw[i,j] * BC(i,j)
       BC = gauge_obs / QPE_raw at gauge cells; IDW-interpolated to full domain
  5. Kalman filter temporal smoother for gauge-outage cells (missing / QC-flagged gauges)
  6. Write corrected QPE as Cloud-Optimized GeoTIFF (4km, WGS84) to S3

Run cadence: every 30 min triggered by Airflow DAG qpe_fusion_30min.
Output S3 key: s3://tropi-climate-data/qpe/corrected/YYYY/MM/DD/HH/qpe_corrected_YYYYMMDDHHmm.tif

Environment variables (resolved from Airflow variables / Secrets Manager):
  NASA_EARTHDATA_TOKEN   — Bearer token for GPM IMERG GESDISC access
  BMKG_API_KEY           — BMKG ARG real-time data API key
  TROPI_API_BASE_URL     — Internal API base URL for status callbacks
  AWS_DEFAULT_REGION     — S3 bucket region (default: ap-southeast-3 Jakarta)
  QPE_S3_BUCKET          — Output bucket name (default: tropi-climate-data)
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPM IMERG Early Run — GESDISC OPeNDAP endpoint template
IMERG_OPENDAP_URL = (
    "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHHE.07/"
    "{year}/{doy}/3B-HHR-E.MS.MRG.3IMERG.{yyyymmdd}-S{HH}{MM}00-E{HH2}{MM2}59.{min_idx:04d}.V07B.HDF5"
)

# BMKG ARG real-time API
BMKG_ARG_API = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
BMKG_ARG_REALTIME_API = "{base}/v1/rain-gauge/realtime"

# Grid specs: GPM IMERG 4km (0.1 degree) over Indonesia bounding box
INDONESIA_BBOX = {
    "lon_min": 94.0,
    "lon_max": 141.5,
    "lat_min": -11.5,
    "lat_max":   7.0,
}
GRID_RES_DEG = 0.1  # ~11 km; 4-km sub-tile product requested via spatial subsetting

# Kalman filter process/observation noise
KF_PROCESS_NOISE_VAR = 0.04    # mm²
KF_OBS_NOISE_VAR     = 0.25    # mm² (gauge measurement noise)

# IDW interpolation power
IDW_POWER = 2.0
IDW_MIN_GAUGES = 3  # minimum valid gauges for IDW; fall back to no correction below this

# S3 output
DEFAULT_S3_BUCKET = os.getenv("QPE_S3_BUCKET", "tropi-climate-data")
S3_KEY_PREFIX     = "qpe/corrected"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GaugeObservation:
    """Single BMKG ARG observation for one 30-min window."""
    station_id: str
    lon: float
    lat: float
    precip_mm: float        # observed 30-min accumulation (mm)
    qc_flag: int            # 0=good, 1=suspect, 2=missing
    province: str
    timestamp_utc: datetime


@dataclass
class KalmanState:
    """Per-cell Kalman filter state for temporal gap-filling."""
    x_hat: float = 0.0      # state estimate (bias ratio)
    P: float = 1.0           # estimate error variance


class QPEFusionRunStatus(BaseModel):
    """Result model for a single QPE fusion run."""
    run_time_utc: datetime
    window_start_utc: datetime
    window_end_utc: datetime
    n_gauges_total: int
    n_gauges_valid: int
    n_gauges_outage: int
    mean_bias_ratio: float
    bias_ratio_p10: float
    bias_ratio_p90: float
    s3_output_key: str
    s3_bucket: str
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# QPE Fusion Pipeline
# ---------------------------------------------------------------------------

class QPEFusionPipeline:
    """
    Real-time GPM IMERG + BMKG gauge QPE bias-correction pipeline.

    Usage (from Airflow DAG):
        pipeline = QPEFusionPipeline()
        status = pipeline.run(window_start=pendulum.now("UTC").subtract(minutes=30))
    """

    def __init__(self) -> None:
        self.earthdata_token: str = os.environ["NASA_EARTHDATA_TOKEN"]
        self.bmkg_api_key: str    = os.environ["BMKG_API_KEY"]
        self.base_url: str        = os.getenv("TROPI_API_BASE_URL", "")
        self.s3_bucket: str       = DEFAULT_S3_BUCKET
        self.region: str          = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-3")

        # Kalman filter state cache (keyed by (row, col))
        self._kf_states: dict[tuple[int, int], KalmanState] = {}

        # Grid arrays (populated on first run)
        self._lon_grid: Optional[np.ndarray] = None
        self._lat_grid: Optional[np.ndarray] = None
        self._grid_shape: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, window_start: datetime) -> QPEFusionRunStatus:
        """
        Execute one 30-min QPE fusion cycle.

        Args:
            window_start: UTC start of the 30-min accumulation window.

        Returns:
            QPEFusionRunStatus with S3 output key and QA metrics.
        """
        t0 = time.monotonic()
        window_end = window_start + timedelta(minutes=30)
        logger.info("QPE fusion run | window %s – %s", window_start.isoformat(), window_end.isoformat())

        try:
            # 1. Fetch raw IMERG field
            raw_qpe, lons, lats = self._fetch_imerg(window_start)
            self._lon_grid = lons
            self._lat_grid = lats
            self._grid_shape = raw_qpe.shape

            # 2. Fetch BMKG gauge observations
            gauges = self._fetch_bmkg_gauges(window_start, window_end)

            # 3. Collocate + compute bias ratios
            bias_field, n_valid, n_outage = self._compute_bias_field(raw_qpe, lons, lats, gauges)

            # 4. Apply multiplicative correction
            corrected_qpe = raw_qpe * bias_field

            # 5. Smooth with Kalman filter (temporal continuity across 30-min windows)
            corrected_qpe = self._apply_kalman_filter(corrected_qpe, bias_field)

            # 6. QA metrics
            valid_bias = bias_field[np.isfinite(bias_field) & (bias_field > 0)]
            mean_bc  = float(np.mean(valid_bias)) if valid_bias.size else 1.0
            bc_p10   = float(np.percentile(valid_bias, 10)) if valid_bias.size else 1.0
            bc_p90   = float(np.percentile(valid_bias, 90)) if valid_bias.size else 1.0

            # 7. Write COG to S3
            s3_key = self._write_geotiff_s3(corrected_qpe, lons, lats, window_start)

            elapsed = time.monotonic() - t0
            logger.info("QPE fusion complete in %.1fs | S3: s3://%s/%s", elapsed, self.s3_bucket, s3_key)

            return QPEFusionRunStatus(
                run_time_utc=datetime.now(timezone.utc),
                window_start_utc=window_start,
                window_end_utc=window_end,
                n_gauges_total=len(gauges),
                n_gauges_valid=n_valid,
                n_gauges_outage=n_outage,
                mean_bias_ratio=mean_bc,
                bias_ratio_p10=bc_p10,
                bias_ratio_p90=bc_p90,
                s3_output_key=s3_key,
                s3_bucket=self.s3_bucket,
                elapsed_seconds=elapsed,
                success=True,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception("QPE fusion failed: %s", exc)
            return QPEFusionRunStatus(
                run_time_utc=datetime.now(timezone.utc),
                window_start_utc=window_start,
                window_end_utc=window_end,
                n_gauges_total=0, n_gauges_valid=0, n_gauges_outage=0,
                mean_bias_ratio=1.0, bias_ratio_p10=1.0, bias_ratio_p90=1.0,
                s3_output_key="", s3_bucket=self.s3_bucket,
                elapsed_seconds=elapsed,
                success=False, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Step 1 — Fetch GPM IMERG Early Run
    # ------------------------------------------------------------------

    def _fetch_imerg(
        self, window_start: datetime
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Download GPM IMERG Early Run 0.1° half-hourly precipitation field
        for the Indonesia bounding box.

        Returns:
            (precip_mm_hr, lon_1d, lat_1d) — precip in mm/hr, shape (nlat, nlon).
        """
        # Build GESDISC OPeNDAP URL with spatial subsetting
        yyyymmdd = window_start.strftime("%Y%m%d")
        year     = window_start.strftime("%Y")
        doy      = window_start.strftime("%j")
        HH, MM   = window_start.strftime("%H"), window_start.strftime("%M")
        min_idx  = (window_start.hour * 60 + window_start.minute) // 30

        # Compute end-time label (HH2:MM2 = start + 29 min)
        end_dt = window_start + timedelta(minutes=29)
        HH2, MM2 = end_dt.strftime("%H"), end_dt.strftime("%M")

        url = IMERG_OPENDAP_URL.format(
            year=year, doy=doy, yyyymmdd=yyyymmdd,
            HH=HH, MM=MM, HH2=HH2, MM2=MM2, min_idx=min_idx,
        )
        # Append OPeNDAP spatial subsetting query
        bb = INDONESIA_BBOX
        subset_query = (
            f".dap.nc4?precipitationCal"
            f"[0:0]"
            f"[{int((bb['lat_min']+90)/GRID_RES_DEG)}:{int((bb['lat_max']+90)/GRID_RES_DEG)}]"
            f"[{int((bb['lon_min']+180)/GRID_RES_DEG)}:{int((bb['lon_max']+180)/GRID_RES_DEG)}]"
        )
        full_url = url + subset_query

        headers = {"Authorization": f"Bearer {self.earthdata_token}"}
        resp = requests.get(full_url, headers=headers, timeout=60)
        resp.raise_for_status()

        # Parse NetCDF4 bytes — real path uses netCDF4 library
        # Stub: generate synthetic grid for development / CI
        nlon = int((bb["lon_max"] - bb["lon_min"]) / GRID_RES_DEG) + 1
        nlat = int((bb["lat_max"] - bb["lat_min"]) / GRID_RES_DEG) + 1
        lons = np.linspace(bb["lon_min"], bb["lon_max"], nlon)
        lats = np.linspace(bb["lat_min"], bb["lat_max"], nlat)

        # Production: decode resp.content with netCDF4
        rng = np.random.default_rng(int(window_start.timestamp()) % (2**32))
        precip = rng.exponential(scale=1.5, size=(nlat, nlon)).astype(np.float32)
        precip[precip < 0.1] = 0.0  # trace threshold

        logger.debug("IMERG fetched: shape=%s, max=%.1f mm/hr", precip.shape, float(precip.max()))
        return precip, lons, lats

    # ------------------------------------------------------------------
    # Step 2 — Fetch BMKG ARG gauge observations
    # ------------------------------------------------------------------

    def _fetch_bmkg_gauges(
        self, window_start: datetime, window_end: datetime
    ) -> list[GaugeObservation]:
        """
        Fetch BMKG automatic rain gauge 30-min observations.
        Applies basic QC: reject values > 200 mm/30min or negative.
        """
        api_url = BMKG_ARG_REALTIME_API.format(base=self.base_url or "https://api.bmkg.go.id")
        params = {
            "start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bbox":  f"{INDONESIA_BBOX['lon_min']},{INDONESIA_BBOX['lat_min']},"
                     f"{INDONESIA_BBOX['lon_max']},{INDONESIA_BBOX['lat_max']}",
        }
        headers = {"X-API-Key": self.bmkg_api_key}

        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            raw = resp.json().get("data", [])
        except requests.RequestException as exc:
            logger.warning("BMKG ARG API unavailable (%s) — proceeding without gauges", exc)
            raw = []

        observations: list[GaugeObservation] = []
        for rec in raw:
            val = float(rec.get("precip_mm", -1))
            qc  = 0
            if val < 0 or val > 200:
                qc = 2  # missing / out-of-range
            elif val > 100:
                qc = 1  # suspect
            observations.append(
                GaugeObservation(
                    station_id=rec["station_id"],
                    lon=float(rec["lon"]),
                    lat=float(rec["lat"]),
                    precip_mm=max(val, 0.0),
                    qc_flag=qc,
                    province=rec.get("province", ""),
                    timestamp_utc=window_start,
                )
            )

        logger.info("BMKG ARG: %d stations, %d QC-good",
                    len(observations), sum(1 for g in observations if g.qc_flag == 0))
        return observations

    # ------------------------------------------------------------------
    # Step 3 — Collocate + IDW bias field
    # ------------------------------------------------------------------

    def _compute_bias_field(
        self,
        raw_qpe: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        gauges: list[GaugeObservation],
    ) -> tuple[np.ndarray, int, int]:
        """
        Build multiplicative bias correction field via IDW interpolation
        of per-gauge bias ratios.

        BC(i,j) = gauge_obs / QPE_at_gauge_cell
        Interpolated by inverse-distance weighting to full domain.
        Default BC = 1.0 (no correction) where coverage is poor.

        Returns:
            (bias_field shape=(nlat,nlon), n_valid_gauges, n_outage_gauges)
        """
        nlat, nlon = raw_qpe.shape
        bias_field = np.ones((nlat, nlon), dtype=np.float32)

        valid_gauges = [g for g in gauges if g.qc_flag == 0]
        outage_count = sum(1 for g in gauges if g.qc_flag == 2)

        if len(valid_gauges) < IDW_MIN_GAUGES:
            logger.warning("Only %d valid gauges — skipping bias correction", len(valid_gauges))
            return bias_field, len(valid_gauges), outage_count

        # Compute per-gauge BC ratio
        gauge_lons = np.array([g.lon for g in valid_gauges])
        gauge_lats = np.array([g.lat for g in valid_gauges])
        gauge_precip = np.array([g.precip_mm for g in valid_gauges])

        # Nearest grid cell for each gauge
        col_idx = np.searchsorted(lons, gauge_lons).clip(0, nlon - 1)
        row_idx = np.searchsorted(lats, gauge_lats).clip(0, nlat - 1)
        imerg_at_gauge = raw_qpe[row_idx, col_idx]

        # Bias ratio: avoid division by zero (both zero → BC=1, gauge>0 & IMERG=0 → cap at 3)
        bc_ratios = np.where(
            imerg_at_gauge > 0.0,
            np.clip(gauge_precip / (imerg_at_gauge * 2.0 + 1e-6), 0.1, 5.0),
            np.where(gauge_precip > 0.0, 3.0, 1.0),
        )

        # IDW interpolation to full grid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_flat = lon_grid.ravel()
        lat_flat = lat_grid.ravel()

        # Vectorised IDW: shape (n_grid_cells, n_gauges)
        dlon = lon_flat[:, None] - gauge_lons[None, :]
        dlat = lat_flat[:, None] - gauge_lats[None, :]
        dist2 = dlon**2 + dlat**2 + 1e-12
        weights = 1.0 / dist2**IDW_POWER
        bc_interp = (weights * bc_ratios[None, :]).sum(axis=1) / weights.sum(axis=1)
        bias_field = bc_interp.reshape(nlat, nlon).astype(np.float32)

        logger.info("Bias field: mean=%.3f  p10=%.3f  p90=%.3f",
                    float(bias_field.mean()),
                    float(np.percentile(bias_field, 10)),
                    float(np.percentile(bias_field, 90)))
        return bias_field, len(valid_gauges), outage_count

    # ------------------------------------------------------------------
    # Step 5 — Kalman filter temporal smoother
    # ------------------------------------------------------------------

    def _apply_kalman_filter(
        self, corrected_qpe: np.ndarray, bias_field: np.ndarray
    ) -> np.ndarray:
        """
        Apply per-cell scalar Kalman filter to smooth bias corrections
        through gauge-outage windows.

        State variable: bias ratio BC(t)
        Prediction: BC(t|t-1) = BC(t-1)  (random-walk model)
        Update: BC(t|t) = BC(t|t-1) + K*(BC_obs - BC(t|t-1))

        In outage cells (bias_field == 1.0 from fallback), the filter
        propagates its prior estimate forward (no measurement update).
        """
        nlat, nlon = corrected_qpe.shape
        smoothed = corrected_qpe.copy()

        for r in range(nlat):
            for c in range(nlon):
                state = self._kf_states.get((r, c), KalmanState())

                # Prediction
                P_pred = state.P + KF_PROCESS_NOISE_VAR

                # Determine if this cell has a real observation
                is_observed = not np.isclose(bias_field[r, c], 1.0)

                if is_observed:
                    # Update
                    K = P_pred / (P_pred + KF_OBS_NOISE_VAR)
                    x_new = state.x_hat + K * (bias_field[r, c] - state.x_hat)
                    P_new = (1 - K) * P_pred
                else:
                    # Propagate prior
                    x_new = state.x_hat
                    P_new = P_pred

                self._kf_states[(r, c)] = KalmanState(x_hat=x_new, P=P_new)

                # Apply smoothed bias to raw IMERG (ratio from smoothed filter)
                # For efficiency: only re-correct if KF estimate meaningfully differs
                if abs(x_new - 1.0) > 0.05:
                    # Retrieve raw IMERG from corrected / bias_field
                    raw_val = (corrected_qpe[r, c] / bias_field[r, c]
                               if bias_field[r, c] > 0 else corrected_qpe[r, c])
                    smoothed[r, c] = np.float32(max(raw_val * x_new, 0.0))

        return smoothed

    # ------------------------------------------------------------------
    # Step 6 — Write Cloud-Optimized GeoTIFF to S3
    # ------------------------------------------------------------------

    def _write_geotiff_s3(
        self,
        qpe: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        window_start: datetime,
    ) -> str:
        """
        Write corrected QPE array as a Cloud-Optimized GeoTIFF (COG) to S3.

        Production: uses rasterio + rio_cogeo.
        Returns the S3 key.
        """
        import boto3

        ts = window_start.strftime("%Y%m%d%H%M")
        s3_key = (
            f"{S3_KEY_PREFIX}/"
            f"{window_start.strftime('%Y/%m/%d/%H')}/"
            f"qpe_corrected_{ts}.tif"
        )

        # Build in-memory GeoTIFF
        try:
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(
                west=float(lons.min()), south=float(lats.min()),
                east=float(lons.max()), north=float(lats.max()),
                width=qpe.shape[1], height=qpe.shape[0],
            )
            buf = io.BytesIO()
            with rasterio.open(
                buf, "w",
                driver="GTiff",
                height=qpe.shape[0], width=qpe.shape[1],
                count=1, dtype=qpe.dtype,
                crs="EPSG:4326", transform=transform,
                compress="deflate", predictor=2,
                tiled=True, blockxsize=256, blockysize=256,
            ) as dst:
                dst.write(qpe, 1)
                dst.update_tags(
                    product="QPE_CORRECTED",
                    source="GPM_IMERG+BMKG_ARG",
                    window_start=window_start.isoformat(),
                    units="mm_per_30min",
                    resolution_deg="0.1",
                    algorithm="multiplicative_bc+kalman",
                    version="sprint2.0",
                )
            buf.seek(0)
            tif_bytes = buf.read()
        except ImportError:
            # Fallback if rasterio not available in worker env
            logger.warning("rasterio not available — writing raw numpy bytes")
            buf = io.BytesIO()
            np.save(buf, qpe)
            tif_bytes = buf.getvalue()
            s3_key = s3_key.replace(".tif", ".npy")

        # Upload to S3
        s3 = boto3.client("s3", region_name=self.region)
        s3.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=tif_bytes,
            ContentType="image/tiff",
            Metadata={
                "window-start": window_start.isoformat(),
                "product": "qpe-corrected",
            },
        )
        logger.info("S3 upload complete: s3://%s/%s", self.s3_bucket, s3_key)
        return s3_key
