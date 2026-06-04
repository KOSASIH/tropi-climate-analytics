"""
Basic API tests — Tropi-Climate-Analytics
Run: pytest src/tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "CLIMATE-OS" in data["agents"]


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Tropi-Climate-Analytics" in resp.json()["message"]


def test_climate_temperature_empty():
    resp = client.get("/api/v1/climate/temperature")
    assert resp.status_code == 200
    assert resp.json() == []


def test_climate_aqi_empty():
    resp = client.get("/api/v1/climate/aqi")
    assert resp.status_code == 200
    assert resp.json() == []


def test_alerts_active_empty():
    resp = client.get("/api/v1/alerts/active")
    assert resp.status_code == 200
    assert resp.json() == []


def test_geospatial_deforestation_empty():
    resp = client.get("/api/v1/geospatial/deforestation")
    assert resp.status_code == 200
    assert resp.json() == []


def test_geospatial_landcover():
    resp = client.get("/api/v1/geospatial/landcover?province=Kalimantan+Timur&year=2024")
    assert resp.status_code == 200
    data = resp.json()
    assert data["region"] == "Kalimantan Timur"
    assert data["year"] == 2024
