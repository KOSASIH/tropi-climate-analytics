"""
Tropi-Climate-Analytics — Main FastAPI Application
CLIMATE-OS | Sprint 0 Bootstrap
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from src.api.routes import alerts, climate, geospatial
from src.data.storage.database import database


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup -> yield -> shutdown"""
    logger.info("Tropi-Climate-Analytics starting up...")
    await database.connect()
    logger.info("Database connected")
    yield
    logger.info("Shutting down...")
    await database.disconnect()


app = FastAPI(
    title="Tropi-Climate-Analytics",
    description=(
        "Cloud-based big-data analytics platform for tropical climate monitoring. "
        "Integrates NASA satellite data (Landsat, MODIS, SMAP, GPM) with Indonesian "
        "ground observations to monitor deforestation, air quality, SST, and rainfall."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(climate.router, prefix="/api/v1", tags=["Climate"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(geospatial.router, prefix="/api/v1", tags=["Geospatial"])


@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Platform health check endpoint"""
    return {
        "status": "healthy",
        "service": "tropi-climate-analytics",
        "version": "0.1.0",
        "agents": [
            "CLIMATE-OS", "DATA-FLOW", "ATMOSPHERE", "GEOSPATIAL",
            "HYDROLOGIS", "ANALYTICA", "TERRA-VISION", "API-GATEWAY",
            "SECURE-GUARD", "CLOUD-FORGE", "COMM-ONNECT",
        ],
    }


@app.get("/", tags=["System"])
async def root() -> dict:
    return {"message": "Tropi-Climate-Analytics API", "docs": "/docs", "health": "/health"}
