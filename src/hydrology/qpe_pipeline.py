"""
qpe_pipeline.py — Sprint 8 J1
QPEPipeline: Merge GPM IMERG half-hourly precipitation (0.1° grid) with
BMKG rain-gauge network via Optimal Interpolation (OI) to produce 4km/30-min QPE rasters.

GPM source:   workspace/data/gpm/IMERG_{YYYYMMDD_HHMM}.nc  (stub: synthetic gaussian field if absent)
Gauge source: workspace/data/bmkg/gauges_{YYYYMMDD}.csv    (cols: station_id, lat, lon, precip_mm_30min)

OI parameters:
  decorrelation length L = 50 km
  observation error σ_o  = 1.5 mm
  background error σ_b   = 2.0 mm

Outputs:
  workspace/output/qpe/qpe_{YYYYMMDD_HHMM}.geojson   — 4km grid cells with precip_mm_30min
  workspace/output/qpe/latest_qpe.json                — sidecar for GEOSPATIAL + emergency pickup

Prometheus: QPE_UPDATE_LATENCY{source=gpm|gauge|merged} Histogram
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OI parameters
# ---------------------------------------------------------------------------
_L_KM       = 50.0     # decorrelation length (km)
_SIGMA_O    = 1.5      # observation error std (mm)
_SIGMA_B    = 2.0      # background error std (mm)
_GRID_RES   = 0.04     # ~4 km in degrees (≈ 0.04°)
_EARTH_R_KM = 6371.0   # Earth radius for haversine

# Indonesia domain
_LON_MIN, _LON_MAX = 95.0, 141.0
_LAT_MIN, _LAT_MAX = -11.0, 6.0

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QPEResult:
    valid_time:         str         # ISO 8601
    num_gauges_merged:  int
    max_precip_mm:      float
    mean_precip_mm:     float
    geojson_path:       str
    sidecar_path:       str
    method:             str         # "oi_merged" | "gpm_only" | "gauge_only" | "synthetic"
    gpm_latency_s:      float
    merge_latency_s:    float
    warnings:           list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    r     = _EARTH_R_KM
    phi1  = math.radians(lat1)
    phi2  = math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a     = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _corr(d_km: float, L: float = _L_KM) -> float:
    """Gaussian decorrelation function."""
    return math.exp(-0.5 * (d_km / L) ** 2)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QPEPipeline:
    """
    QPE generator: merges GPM IMERG background field with BMKG gauge network
    using Optimal Interpolation (Gandin 1965 formulation).
    """

    WORKSPACE  = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    GPM_DIR    = WORKSPACE / "data" / "gpm"
    BMKG_DIR   = WORKSPACE / "data" / "bmkg"
    OUTPUT_DIR = WORKSPACE / "output" / "qpe"
    SIDECAR    = WORKSPACE / "output" / "qpe" / "latest_qpe.json"
    QUEUE      = WORKSPACE / "output" / "qpe" / "qpe_queue.jsonl"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, valid_time: datetime) -> QPEResult:
        """
        Run QPE pipeline for *valid_time*.
        1. Load/stub GPM IMERG background field
        2. Load BMKG gauge observations
        3. Apply Optimal Interpolation correction
        4. Write GeoJSON + sidecar
        Returns QPEResult.
        """
        t0       = time.monotonic()
        warnings: list[str] = []

        ts_str = valid_time.strftime("%Y%m%d_%H%M")
        date_str = valid_time.strftime("%Y%m%d")

        # --- Step 1: GPM background ---
        t_gpm0 = time.monotonic()
        background, gpm_source = self._load_gpm(valid_time, ts_str, warnings)
        gpm_lat_s = time.monotonic() - t_gpm0
        self._emit_latency("gpm", gpm_lat_s)

        # --- Step 2: BMKG gauges ---
        gauges = self._load_gauges(date_str, warnings)
        self._emit_latency("gauge", time.monotonic() - t0 - gpm_lat_s)

        # --- Step 3: OI merge ---
        t_oi0 = time.monotonic()
        if gauges:
            merged, method = self._optimal_interpolation(background, gauges)
        else:
            merged, method = background, gpm_source
            warnings.append("No gauge observations; using GPM background without OI correction")
        merge_lat_s = time.monotonic() - t_oi0
        self._emit_latency("merged", merge_lat_s)

        # --- Step 4: Stats ---
        all_vals   = [v for row in merged.values() for v in row.values()]
        max_precip = max(all_vals) if all_vals else 0.0
        mean_prec  = sum(all_vals) / len(all_vals) if all_vals else 0.0

        # --- Step 5: Write GeoJSON ---
        geo_path = self.OUTPUT_DIR / f"qpe_{ts_str}.geojson"
        geojson  = self._build_geojson(merged, valid_time.isoformat(), method)
        with open(geo_path, "w") as f:
            json.dump(geojson, f, indent=2)

        # --- Step 6: Sidecar ---
        sidecar = {
            "valid_time":        valid_time.isoformat(),
            "max_precip_mm":     round(max_precip, 3),
            "mean_precip_mm":    round(mean_prec, 3),
            "num_gauges_merged": len(gauges),
            "method":            method,
            "geojson_path":      str(geo_path),
            "gpm_latency_s":     round(gpm_lat_s, 3),
            "merge_latency_s":   round(merge_lat_s, 3),
        }
        with open(self.SIDECAR, "w") as f:
            json.dump(sidecar, f, indent=2)

        result = QPEResult(
            valid_time         = valid_time.isoformat(),
            num_gauges_merged  = len(gauges),
            max_precip_mm      = round(max_precip, 3),
            mean_precip_mm     = round(mean_prec, 3),
            geojson_path       = str(geo_path),
            sidecar_path       = str(self.SIDECAR),
            method             = method,
            gpm_latency_s      = round(gpm_lat_s, 3),
            merge_latency_s    = round(merge_lat_s, 3),
            warnings           = warnings,
        )

        logger.info(
            "QPE | %s method=%s gauges=%d max=%.1f mm mean=%.2f mm",
            ts_str, method, len(gauges), max_precip, mean_prec,
        )
        return result

    # ------------------------------------------------------------------
    # Private: data loaders
    # ------------------------------------------------------------------

    def _load_gpm(
        self,
        valid_time: datetime,
        ts_str:     str,
        warnings:   list[str],
    ) -> tuple[dict[str, dict[str, float]], str]:
        """
        Load GPM IMERG .nc file; return dict[lat_idx][lon_idx] → precip_mm.
        Falls back to synthetic Gaussian field if absent.
        """
        nc_path = self.GPM_DIR / f"IMERG_{ts_str}.nc"
        if nc_path.exists():
            try:
                return self._parse_imerg_nc(nc_path), "gpm_imerg"
            except Exception as exc:
                warnings.append(f"GPM parse error ({exc}); using synthetic background")

        warnings.append(f"GPM IMERG file absent ({nc_path}); generating synthetic field")
        return self._synthetic_gpm(valid_time), "synthetic"

    @staticmethod
    def _parse_imerg_nc(nc_path: Path) -> dict[str, dict[str, float]]:
        """Stub: parse IMERG NetCDF. In production use netCDF4 or xarray."""
        # When the real file exists, load precipitationCal variable
        # For now, treat as if it returns an empty dict → caller falls back
        raise NotImplementedError("Real IMERG NetCDF parser not yet implemented — use synthetic fallback")

    @staticmethod
    def _synthetic_gpm(valid_time: datetime) -> dict[str, dict[str, float]]:
        """
        Synthetic Gaussian precipitation field over Indonesia domain.
        Used when GPM files are unavailable (stub / test environment).
        Seed from valid_time for deterministic but time-varying output.
        """
        import random
        rng = random.Random(int(valid_time.timestamp()))

        # Place 3-5 precipitation cells
        n_cells = rng.randint(3, 5)
        cells   = [
            {
                "lat": rng.uniform(_LAT_MIN + 1, _LAT_MAX - 1),
                "lon": rng.uniform(_LON_MIN + 5, _LON_MAX - 5),
                "peak": rng.uniform(1.0, 25.0),
                "r_km": rng.uniform(30.0, 150.0),
            }
            for _ in range(n_cells)
        ]

        background: dict[str, dict[str, float]] = {}
        lat = _LAT_MIN
        while lat <= _LAT_MAX:
            lon = _LON_MIN
            background[f"{lat:.2f}"] = {}
            while lon <= _LON_MAX:
                val = 0.0
                for c in cells:
                    d   = _haversine_km(lat, lon, c["lat"], c["lon"])
                    val += c["peak"] * math.exp(-0.5 * (d / c["r_km"]) ** 2)
                background[f"{lat:.2f}"][f"{lon:.2f}"] = max(round(val, 3), 0.0)
                lon += _GRID_RES
            lat += _GRID_RES

        return background

    def _load_gauges(
        self,
        date_str: str,
        warnings: list[str],
    ) -> list[dict[str, float]]:
        """
        Load BMKG gauge observations from CSV.
        Returns list of dicts: {station_id, lat, lon, precip_mm_30min}.
        """
        csv_path = self.BMKG_DIR / f"gauges_{date_str}.csv"
        if not csv_path.exists():
            warnings.append(f"BMKG gauge file absent ({csv_path}); no gauge correction")
            return []

        gauges = []
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        gauges.append({
                            "station_id":      row["station_id"],
                            "lat":             float(row["lat"]),
                            "lon":             float(row["lon"]),
                            "precip_mm_30min": max(float(row["precip_mm_30min"]), 0.0),
                        })
                    except (KeyError, ValueError):
                        pass
        except Exception as exc:
            warnings.append(f"BMKG gauge parse error ({exc})")

        logger.info("Loaded %d BMKG gauge observations", len(gauges))
        return gauges

    # ------------------------------------------------------------------
    # Private: OI core
    # ------------------------------------------------------------------

    def _optimal_interpolation(
        self,
        background: dict[str, dict[str, float]],
        gauges:     list[dict[str, float]],
    ) -> tuple[dict[str, dict[str, float]], str]:
        """
        Optimal Interpolation (OI) correction of background field using gauge obs.

        Analysis increment: Δa = B_og · (B_gg + R)^-1 · (y_o - y_b)
          B_gg: gauge-gauge background error covariance (σ_b² × corr(d_ij))
          R:    observation error covariance (σ_o² · I)
          B_og: grid-gauge background error covariance (σ_b² × corr(d_ig))
          y_o:  gauge observations
          y_b:  background at gauge locations

        For large gauge networks, uses diagonal-plus-local approximation
        (only gauges within 3·L are used per grid point).
        """
        n = len(gauges)
        if n == 0:
            return background, "gpm_only"

        # Build B_gg + R matrix and innovation vector once
        B_gg_R = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                d   = _haversine_km(gauges[i]["lat"], gauges[i]["lon"],
                                    gauges[j]["lat"], gauges[j]["lon"])
                cov = (_SIGMA_B ** 2) * _corr(d)
                B_gg_R[i][j] = cov + ((_SIGMA_O ** 2) if i == j else 0.0)

        # Invert B_gg_R (small matrix — direct Gauss-Jordan)
        B_inv = _mat_inv(B_gg_R)

        # Grid background interpolation at gauge locations
        y_b = []
        for g in gauges:
            glat = f"{round(round(g['lat'] / _GRID_RES) * _GRID_RES, 2):.2f}"
            glon = f"{round(round(g['lon'] / _GRID_RES) * _GRID_RES, 2):.2f}"
            y_b.append(background.get(glat, {}).get(glon, 0.0))

        innovations = [gauges[i]["precip_mm_30min"] - y_b[i] for i in range(n)]

        # Apply OI correction to each grid point (localize to 3L radius)
        merged = {}
        radius = 3.0 * _L_KM
        for lat_k, lon_row in background.items():
            merged[lat_k] = {}
            lat_f = float(lat_k)
            for lon_k, bg_val in lon_row.items():
                lon_f = float(lon_k)
                # Local gauges within 3L
                local_idx = [
                    i for i, g in enumerate(gauges)
                    if _haversine_km(lat_f, lon_f, g["lat"], g["lon"]) <= radius
                ]
                if not local_idx:
                    merged[lat_k][lon_k] = bg_val
                    continue

                # B_og for local gauges
                B_og = []
                for i in local_idx:
                    d   = _haversine_km(lat_f, lon_f, gauges[i]["lat"], gauges[i]["lon"])
                    B_og.append((_SIGMA_B ** 2) * _corr(d))

                # Sub-matrix of B_inv
                B_inv_local = [[B_inv[i][j] for j in local_idx] for i in local_idx]
                innov_local = [innovations[i] for i in local_idx]

                # Δa = B_og · B_inv_local · innov_local
                Bw     = _mat_vec(B_inv_local, innov_local)
                delta_a = sum(B_og[k] * Bw[k] for k in range(len(local_idx)))
                merged[lat_k][lon_k] = max(bg_val + delta_a, 0.0)

        return merged, "oi_merged"

    # ------------------------------------------------------------------
    # Private: output builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_geojson(
        grid:       dict[str, dict[str, float]],
        valid_time: str,
        method:     str,
    ) -> dict[str, Any]:
        """Build GeoJSON FeatureCollection of 4km grid cells."""
        features = []
        half = _GRID_RES / 2.0
        for lat_k, lon_row in grid.items():
            lat = float(lat_k)
            for lon_k, precip in lon_row.items():
                if precip < 0.01:
                    continue  # omit dry cells to keep file size manageable
                lon = float(lon_k)
                coords = [[
                    [lon - half, lat - half],
                    [lon + half, lat - half],
                    [lon + half, lat + half],
                    [lon - half, lat + half],
                    [lon - half, lat - half],
                ]]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": coords},
                    "properties": {
                        "precip_mm_30min": round(precip, 3),
                        "valid_time":      valid_time,
                        "method":          method,
                    },
                })
        return {"type": "FeatureCollection", "features": features}

    @staticmethod
    def _emit_latency(source: str, latency_s: float) -> None:
        try:
            from src.hydrology.metrics import QPE_UPDATE_LATENCY
            QPE_UPDATE_LATENCY.labels(source=source).observe(latency_s)
        except Exception as exc:
            logger.debug("QPE_UPDATE_LATENCY unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Tiny linear-algebra helpers (avoid numpy dependency in stub)
# ---------------------------------------------------------------------------

def _mat_inv(A: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan matrix inverse for small n × n matrices."""
    n  = len(A)
    AM = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(AM[r][col]))
        AM[col], AM[pivot] = AM[pivot], AM[col]
        div = AM[col][col]
        if abs(div) < 1e-12:
            # Singular; return identity as fallback
            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        AM[col] = [v / div for v in AM[col]]
        for row in range(n):
            if row != col:
                f  = AM[row][col]
                AM[row] = [AM[row][k] - f * AM[col][k] for k in range(2 * n)]
    return [AM[i][n:] for i in range(n)]


def _mat_vec(A: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector product."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]
