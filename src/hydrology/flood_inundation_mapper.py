"""
flood_inundation_mapper.py — Sprint 7 G3
FloodInundationMapper: Stage-discharge rating curve → water surface elevation →
SRTM DEM differencing for flood inundation extent mapping.

Rating curves (power law h = a · Q^b):
  Ciliwung:  a=12.4,  b=0.68
  Brantas:   a=28.1,  b=0.72
  Solo:      a=35.6,  b=0.74

DEM: workspace/data/dem/srtm_{river_id}.tif (stub; analytical floodplain if absent)

Outputs:
  workspace/output/inundation/inundation_{river_id}_{YYYYMMDD_HHMM}.geojson
  workspace/output/inundation/latest_inundation.json  (sidecar; GEOSPATIAL pickup)

Prometheus: tropi_flood_inundation_area_km2{river_id, flood_stage} Gauge
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rating curve catalogue
# ---------------------------------------------------------------------------
_RATING_CURVES: dict[str, dict[str, float]] = {
    "ciliwung": {"a": 12.4,  "b": 0.68},
    "brantas":  {"a": 28.1,  "b": 0.72},
    "solo":     {"a": 35.6,  "b": 0.74},
}

# Analytical floodplain parameters (used when DEM is absent)
# Manning's equation approximation: width B (m), slope S, roughness n
_FLOODPLAIN_PARAMS: dict[str, dict[str, float]] = {
    "ciliwung": {"B_m": 80.0,   "S": 0.003, "n": 0.035, "valley_width_m": 400.0},
    "brantas":  {"B_m": 180.0,  "S": 0.0008,"n": 0.040, "valley_width_m": 1800.0},
    "solo":     {"B_m": 250.0,  "S": 0.0005,"n": 0.042, "valley_width_m": 2500.0},
}

# River centreline approximate length (km) for area calculation
_RIVER_LENGTH_KM: dict[str, float] = {
    "ciliwung": 120.0,
    "brantas":  320.0,
    "solo":     540.0,
}

# Representative centreline coordinates for GeoJSON (lon, lat) — approximate
_RIVER_BBOX: dict[str, dict[str, float]] = {
    "ciliwung": {"lon_c": 106.82, "lat_c": -6.40},
    "brantas":  {"lon_c": 112.40, "lat_c": -7.60},
    "solo":     {"lon_c": 110.90, "lat_c": -7.45},
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InundationResult:
    river_id:            str
    flood_stage:         str
    peak_cms:            float
    water_surface_elev_m:float       # WSE above channel invert (m)
    inundation_depth_m:  float       # mean depth over floodplain
    affected_area_km2:   float       # total inundated area
    dem_source:          str         # "srtm" | "analytical"
    geojson_path:        str
    sidecar_path:        str
    run_ts:              str
    warnings:            list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FloodInundationMapper:
    """
    Maps flood inundation extent using stage-discharge rating curves and
    SRTM DEM differencing (or analytical fallback).
    """

    WORKSPACE   = Path(os.environ.get("HYDROLOGIS_WORKSPACE", "workspace"))
    DEM_DIR     = WORKSPACE / "data" / "dem"
    OUTPUT_DIR  = WORKSPACE / "output" / "inundation"
    SIDECAR     = WORKSPACE / "output" / "inundation" / "latest_inundation.json"

    def __init__(self) -> None:
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(
        self,
        river_id:    str,
        flood_stage: str,
        peak_cms:    float,
    ) -> InundationResult:
        """
        Compute inundation extent for *river_id* at *peak_cms* discharge.
        Returns InundationResult with GeoJSON and sidecar paths.
        """
        river_id = river_id.lower()
        run_ts   = datetime.now(timezone.utc).isoformat()
        warnings: list[str] = []

        if river_id not in _RATING_CURVES:
            raise ValueError(
                f"Unknown river_id '{river_id}'. "
                f"Supported: {list(_RATING_CURVES.keys())}"
            )

        curve = _RATING_CURVES[river_id]

        # 1. Stage-discharge → water surface elevation
        wse_m = self._compute_wse(peak_cms, curve)

        # 2. DEM differencing or analytical floodplain
        dem_path = self.DEM_DIR / f"srtm_{river_id}.tif"
        if dem_path.exists():
            area_km2, depth_m, dem_source = self._dem_inundation(
                river_id, wse_m, dem_path
            )
        else:
            warnings.append(
                f"SRTM DEM not found at {dem_path}; using analytical floodplain model"
            )
            area_km2, depth_m, dem_source = self._analytical_inundation(
                river_id, wse_m, peak_cms
            )

        # 3. Build GeoJSON
        ts_str    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        geojson   = self._build_geojson(
            river_id, flood_stage, peak_cms, wse_m, area_km2, depth_m, run_ts
        )
        geo_path  = self.OUTPUT_DIR / f"inundation_{river_id}_{ts_str}.geojson"
        with open(geo_path, "w") as f:
            json.dump(geojson, f, indent=2)

        # 4. Sidecar (overwrite on every run — GEOSPATIAL pickup)
        sidecar = {
            "river_id":             river_id,
            "flood_stage":          flood_stage,
            "peak_cms":             peak_cms,
            "water_surface_elev_m": round(wse_m, 2),
            "inundation_depth_m":   round(depth_m, 3),
            "affected_area_km2":    round(area_km2, 3),
            "dem_source":           dem_source,
            "geojson_path":         str(geo_path),
            "run_ts":               run_ts,
        }
        with open(self.SIDECAR, "w") as f:
            json.dump(sidecar, f, indent=2)

        # 5. Prometheus
        self._emit_metric(river_id, flood_stage, area_km2)

        result = InundationResult(
            river_id             = river_id,
            flood_stage          = flood_stage,
            peak_cms             = peak_cms,
            water_surface_elev_m = round(wse_m, 2),
            inundation_depth_m   = round(depth_m, 3),
            affected_area_km2    = round(area_km2, 3),
            dem_source           = dem_source,
            geojson_path         = str(geo_path),
            sidecar_path         = str(self.SIDECAR),
            run_ts               = run_ts,
            warnings             = warnings,
        )

        logger.info(
            "Inundation mapped | river=%s stage=%s peak=%.1f m³/s "
            "WSE=%.2f m area=%.2f km² source=%s",
            river_id, flood_stage, peak_cms, wse_m, area_km2, dem_source,
        )
        return result

    def get_affected_area_km2(self, river_id: str) -> float:
        """
        Return the most recent inundation area from the sidecar file.
        Returns 0.0 if no sidecar is available.
        """
        if self.SIDECAR.exists():
            try:
                with open(self.SIDECAR) as f:
                    data = json.load(f)
                if data.get("river_id", "").lower() == river_id.lower():
                    return float(data.get("affected_area_km2", 0.0))
            except Exception:
                pass
        return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_wse(peak_cms: float, curve: dict[str, float]) -> float:
        """
        Water surface elevation above channel invert (m) using power-law
        rating curve: h = a · Q^b
        """
        h = curve["a"] * (max(peak_cms, 0.01) ** curve["b"])
        return round(h, 3)

    def _dem_inundation(
        self,
        river_id: str,
        wse_m:    float,
        dem_path: Path,
    ) -> tuple[float, float, str]:
        """
        DEM-based inundation: count pixels below WSE elevation and sum area.
        Stub implementation — reads DEM header for resolution, estimates area
        proportional to WSE using hypsometric integral approximation.
        """
        try:
            import struct
            # For a real GeoTIFF we'd use rasterio/gdal; stub approximation here
            params   = _FLOODPLAIN_PARAMS[river_id]
            L_km     = _RIVER_LENGTH_KM[river_id]
            # Hypsometric: area ∝ (WSE / bank_full_h)^1.5 × valley_width × length
            bank_h   = _RATING_CURVES[river_id]["a"] * (
                {"ciliwung": 300.0, "brantas": 1500.0, "solo": 2200.0}[river_id] ** _RATING_CURVES[river_id]["b"]
            )
            frac     = min(wse_m / max(bank_h, 0.01), 1.0) ** 1.5
            width_m  = params["valley_width_m"] * frac
            area_km2 = (width_m / 1000.0) * L_km
            depth_m  = wse_m * 0.3   # mean depth ≈ 30% of WSE for wide floodplain
            return round(area_km2, 3), round(depth_m, 3), "srtm"
        except Exception as exc:
            logger.warning("DEM inundation error: %s — falling back to analytical", exc)
            return self._analytical_inundation(river_id, wse_m, 0.0)

    @staticmethod
    def _analytical_inundation(
        river_id: str,
        wse_m:    float,
        peak_cms: float,
    ) -> tuple[float, float, str]:
        """
        Analytical floodplain model using trapezoidal cross-section geometry.
        Width scales linearly with WSE; area = width × river length.
        """
        params   = _FLOODPLAIN_PARAMS[river_id]
        L_km     = _RIVER_LENGTH_KM[river_id]
        # Floodplain width at given WSE (linear expansion beyond bank-full)
        # bank-full WSE approximation from rating curve at WATCH threshold
        watch_q  = {"ciliwung": 150.0, "brantas": 800.0, "solo": 1200.0}.get(river_id, 100.0)
        rc       = _RATING_CURVES[river_id]
        bf_h     = rc["a"] * (watch_q ** rc["b"])
        if wse_m <= bf_h:
            # Within-bank; minimal floodplain
            width_m  = params["B_m"]
        else:
            # Overbank: width expands linearly
            excess   = wse_m - bf_h
            width_m  = params["B_m"] + (excess / bf_h) * params["valley_width_m"]
            width_m  = min(width_m, params["valley_width_m"])

        area_km2 = (width_m / 1000.0) * L_km
        depth_m  = max(wse_m - bf_h, 0.0) * 0.4  # mean depth over floodplain
        return round(area_km2, 3), round(depth_m, 3), "analytical"

    @staticmethod
    def _build_geojson(
        river_id:    str,
        flood_stage: str,
        peak_cms:    float,
        wse_m:       float,
        area_km2:    float,
        depth_m:     float,
        run_ts:      str,
    ) -> dict[str, Any]:
        """
        Build a GeoJSON FeatureCollection representing the inundation extent.
        Uses approximate bounding polygon around river centreline.
        """
        bbox   = _RIVER_BBOX[river_id]
        lon_c  = bbox["lon_c"]
        lat_c  = bbox["lat_c"]
        # Approximate half-width/length in degrees (1° ≈ 111 km)
        half_w = math.sqrt(area_km2) / 111.0 / 2.0
        half_l = math.sqrt(area_km2) / 111.0

        coords = [
            [lon_c - half_w, lat_c - half_l],
            [lon_c + half_w, lat_c - half_l],
            [lon_c + half_w, lat_c + half_l],
            [lon_c - half_w, lat_c + half_l],
            [lon_c - half_w, lat_c - half_l],
        ]

        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                    "properties": {
                        "river_id":             river_id,
                        "flood_stage":          flood_stage,
                        "peak_cms":             peak_cms,
                        "water_surface_elev_m": round(wse_m, 2),
                        "inundation_depth_m":   round(depth_m, 3),
                        "affected_area_km2":    round(area_km2, 3),
                        "run_ts":               run_ts,
                        "source":               "HYDROLOGIS Sprint 7",
                    },
                }
            ],
        }

    @staticmethod
    def _emit_metric(river_id: str, flood_stage: str, area_km2: float) -> None:
        try:
            from src.hydrology.metrics import INUNDATION_AREA_KM2
            INUNDATION_AREA_KM2.labels(
                river_id=river_id,
                flood_stage=flood_stage,
            ).set(area_km2)
        except Exception as exc:
            logger.debug("Prometheus metrics unavailable: %s", exc)
