#!/usr/bin/env python3
"""
Database Initialization Script
Creates PostgreSQL database with PostGIS extension and all tables.
Run: python scripts/setup/initialize_db.py
"""

import os
import sys

import psycopg2
from loguru import logger


def get_connection():
    db_url = os.getenv("DATABASE_URL", "postgresql://tropi_user:password@localhost:5432/tropiclimate")
    # Parse psycopg2 connection from SQLAlchemy-style URL
    url = db_url.replace("postgresql://", "")
    user_pass, host_db = url.split("@")
    user, password = user_pass.split(":")
    host_port, dbname = host_db.split("/")
    host = host_port.split(":")[0]
    port = int(host_port.split(":")[1]) if ":" in host_port else 5432
    return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)


def initialize_database():
    logger.info("Initializing Tropi-Climate-Analytics database...")
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # Enable PostGIS
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    logger.info("PostGIS extensions enabled")

    # Climate data table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS climate_observations (
            id BIGSERIAL PRIMARY KEY,
            dataset VARCHAR(100) NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            location GEOGRAPHY(POINT, 4326),
            value DOUBLE PRECISION,
            unit VARCHAR(20),
            quality_flag INTEGER DEFAULT 0,
            source VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Deforestation events table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS deforestation_events (
            id BIGSERIAL PRIMARY KEY,
            event_id VARCHAR(100) UNIQUE NOT NULL,
            detected_at DATE NOT NULL,
            geom GEOMETRY(POLYGON, 4326),
            area_ha DOUBLE PRECISION,
            province VARCHAR(100),
            district VARCHAR(100),
            confidence FLOAT,
            satellite_source VARCHAR(50),
            previous_cover VARCHAR(50),
            current_cover VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Alerts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS climate_alerts (
            id BIGSERIAL PRIMARY KEY,
            alert_id VARCHAR(100) UNIQUE NOT NULL,
            alert_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            title VARCHAR(255),
            description TEXT,
            affected_area VARCHAR(255),
            location GEOGRAPHY(POINT, 4326),
            radius_km FLOAT,
            issued_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ,
            source_agent VARCHAR(50),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Spatial indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_climate_obs_location ON climate_observations USING GIST(location);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_deforestation_geom ON deforestation_events USING GIST(geom);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_climate_obs_time ON climate_observations(observed_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_active ON climate_alerts(is_active, issued_at);")

    logger.success("Database initialized successfully")
    cur.close()
    conn.close()


if __name__ == "__main__":
    initialize_database()
