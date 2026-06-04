"""
Database Layer — PostgreSQL/PostGIS + Redis cache
Agent: API-GATEWAY / DATA-FLOW
Uses databases[asyncpg] for async PostgreSQL and aioredis for Redis.
"""

import os
from typing import Optional

from loguru import logger


class DatabaseManager:
    """Async database connection manager for PostgreSQL/PostGIS."""

    def __init__(self) -> None:
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql://tropi_user:password@localhost:5432/tropiclimate"
        )
        self._connection = None

    async def connect(self) -> None:
        """Initialize async database connection pool."""
        try:
            # Lazy import to avoid startup error when DB not available in test
            import databases
            self._connection = databases.Database(self.database_url)
            await self._connection.connect()
            logger.info(f"DB connected: {self.database_url.split('@')[-1]}")
        except ImportError:
            logger.warning("databases package not installed — DB unavailable")
        except Exception as e:
            logger.error(f"DB connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        if self._connection:
            await self._connection.disconnect()
            logger.info("DB disconnected")

    async def execute(self, query: str, values: Optional[dict] = None) -> None:
        if self._connection:
            await self._connection.execute(query=query, values=values)

    async def fetch_all(self, query: str, values: Optional[dict] = None) -> list:
        if self._connection:
            return await self._connection.fetch_all(query=query, values=values)
        return []


class CacheManager:
    """Redis cache manager for API response caching and session storage."""

    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL", "3600"))
        self._client = None

    async def connect(self) -> None:
        try:
            import aioredis
            self._client = aioredis.from_url(self.redis_url)
            logger.info("Redis connected")
        except ImportError:
            logger.warning("aioredis not installed — cache unavailable")

    async def get(self, key: str) -> Optional[str]:
        if self._client:
            return await self._client.get(key)
        return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if self._client:
            await self._client.setex(key, ttl or self.default_ttl, value)

    async def delete(self, key: str) -> None:
        if self._client:
            await self._client.delete(key)


# Singleton instances
database = DatabaseManager()
cache = CacheManager()
