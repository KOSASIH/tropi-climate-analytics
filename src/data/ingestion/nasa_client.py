"""
NASA Earth Data CMR API Client — Agent: DATA-FLOW
Supports: Landsat 8/9, MODIS Terra/Aqua, SMAP, GPM IMERG, VIIRS
"""

import os
from datetime import date, datetime
from typing import Optional

import requests
from loguru import logger
from pydantic import BaseModel

CMR_BASE_URL = os.getenv("NASA_CMR_BASE_URL", "https://cmr.earthdata.nasa.gov")
CMR_SEARCH_URL = f"{CMR_BASE_URL}/search"

DATASET_IDS = {
    "MODIS_Terra_SST": "C2036882064-POCLOUD",
    "MODIS_Aqua_SST": "C2036882064-POCLOUD",
    "Landsat8_OLI": "C2021957295-LPCLOUD",
    "Landsat9_OLI2": "C2076090826-LPCLOUD",
    "SMAP_L3_SM": "C1931665183-NSIDC_ECS",
    "GPM_IMERG_HHR": "C2723754864-GES_DISC",
    "MODIS_Terra_NDVI": "C194001210-LPDAAC_ECS",
    "MODIS_Fire": "C1693426528-LPDAAC_ECS",
}

INDONESIA_BBOX = {"lat_min": -11.0, "lat_max": 6.0, "lon_min": 95.0, "lon_max": 141.0}


class GranuleResult(BaseModel):
    granule_id: str
    short_name: str
    time_start: datetime
    time_end: datetime
    download_url: str
    file_size_mb: float
    cloud_cover: Optional[float] = None


class NASACMRClient:
    """Client for NASA CMR granule search over Indonesia bounding box."""

    def __init__(self) -> None:
        self.api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.session = requests.Session()
        user = os.getenv("NASA_EARTHDATA_USERNAME", "")
        pwd = os.getenv("NASA_EARTHDATA_PASSWORD", "")
        if user:
            self.session.auth = (user, pwd)

    def search_granules(
        self,
        dataset: str,
        date_from: date,
        date_to: date,
        bbox: Optional[dict] = None,
        max_results: int = 100,
    ) -> list[GranuleResult]:
        """Search CMR for satellite granules over Indonesia."""
        if bbox is None:
            bbox = INDONESIA_BBOX
        concept_id = DATASET_IDS.get(dataset)
        if not concept_id:
            raise ValueError(f"Unknown dataset: {dataset}")

        params = {
            "concept_id": concept_id,
            "temporal": f"{date_from}T00:00:00Z,{date_to}T23:59:59Z",
            "bounding_box": (
                f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"
            ),
            "page_size": min(max_results, 2000),
            "sort_key": "-start_date",
        }
        resp = self.session.get(f"{CMR_SEARCH_URL}/granules.json", params=params, timeout=30)
        resp.raise_for_status()
        entries = resp.json().get("feed", {}).get("entry", [])
        logger.info(f"CMR: {dataset} | {len(entries)} granules")
        return self._parse_entries(entries)

    def _parse_entries(self, entries: list) -> list[GranuleResult]:
        results = []
        for e in entries:
            links = e.get("links", [])
            url = next(
                (l["href"] for l in links if l.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"),
                "",
            )
            try:
                results.append(GranuleResult(
                    granule_id=e.get("id", ""),
                    short_name=e.get("short_name", ""),
                    time_start=e.get("time_start", "2000-01-01T00:00:00Z"),
                    time_end=e.get("time_end", "2000-01-01T00:00:00Z"),
                    download_url=url,
                    file_size_mb=round(e.get("granule_size", 0), 2),
                    cloud_cover=e.get("cloud_cover"),
                ))
            except Exception as ex:
                logger.warning(f"Parse error: {ex}")
        return results
