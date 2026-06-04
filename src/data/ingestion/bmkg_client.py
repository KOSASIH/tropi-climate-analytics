"""
BMKG Data Client — Agent: DATA-FLOW
Fetches Indonesian meteorological ground observations for satellite bias-correction.
API: https://data.bmkg.go.id
"""

import os
from datetime import date
from typing import Optional

import requests
from loguru import logger
from pydantic import BaseModel

BMKG_BASE_URL = os.getenv("BMKG_BASE_URL", "https://data.bmkg.go.id")


class WeatherObservation(BaseModel):
    station_id: str
    station_name: str
    province: str
    latitude: float
    longitude: float
    elevation_m: float
    observed_at: str
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_mps: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    rainfall_mm: Optional[float] = None
    pressure_hpa: Optional[float] = None


class BMKGClient:
    """Client for BMKG open data API. Normalizes to WeatherObservation schema."""

    def __init__(self) -> None:
        self.session = requests.Session()
        api_key = os.getenv("BMKG_API_KEY", "")
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def get_weather_observations(
        self, date_obs: date, province: Optional[str] = None
    ) -> list[WeatherObservation]:
        """Fetch surface obs from ~150 BMKG AWS stations nationwide."""
        params = {"date": date_obs.isoformat()}
        if province:
            params["province"] = province
        try:
            resp = self.session.get(
                f"{BMKG_BASE_URL}/DataMKG/MEWS/DigitalForecast",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            return self._parse_observations(resp.json())
        except requests.RequestException as e:
            logger.error(f"BMKG error: {e}")
            return []

    def get_rainfall_24h(self, station_id: Optional[str] = None) -> list[dict]:
        """24-hour accumulated rainfall for GPM IMERG bias correction."""
        try:
            resp = self.session.get(
                f"{BMKG_BASE_URL}/DataMKG/MEWS/Curah_Hujan", timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            if station_id:
                data = [d for d in data if d.get("station_id") == station_id]
            return data
        except requests.RequestException as e:
            logger.error(f"BMKG rainfall error: {e}")
            return []

    def _parse_observations(self, raw: dict) -> list[WeatherObservation]:
        results = []
        for item in raw.get("data", []):
            try:
                results.append(WeatherObservation(
                    station_id=item.get("station_id", ""),
                    station_name=item.get("station_name", ""),
                    province=item.get("province", ""),
                    latitude=float(item.get("lat", 0)),
                    longitude=float(item.get("lon", 0)),
                    elevation_m=float(item.get("elevation", 0)),
                    observed_at=item.get("datetime", ""),
                    temperature_c=item.get("temperature"),
                    humidity_pct=item.get("humidity"),
                    wind_speed_mps=item.get("wind_speed"),
                    wind_direction_deg=item.get("wind_dir"),
                    rainfall_mm=item.get("rainfall"),
                    pressure_hpa=item.get("pressure"),
                ))
            except Exception as ex:
                logger.warning(f"BMKG parse error: {ex}")
        return results
