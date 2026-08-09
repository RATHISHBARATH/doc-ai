# ============================================================
# DOC AI DOCA Service – Memory Client
# ============================================================

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
import psycopg2
from psycopg2 import pool

from src.common.config import get_config, DOCAConfig

logger = logging.getLogger(__name__)


class MemoryClient:
    """
    Unified client for short‑term (Redis), long‑term (PostgreSQL), and
    retrieval (vector/graph) memory systems.
    """

    def __init__(self, config: Optional[DOCAConfig] = None):
        if config is None:
            config = get_config()
        self.config = config

        # Redis client for short‑term cache
        self.redis_client = None
        self._init_redis()

        # PostgreSQL connection pool for long‑term storage
        self.pg_pool = None
        self._init_postgres()

    def _init_redis(self) -> None:
        """Initialize Redis client."""
        try:
            # Use environment variable or default from config
            redis_url = self.config.redis_url if hasattr(self.config, 'redis_url') else "redis://redis:6379"
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            # Use environment variable or default from config
            pg_url = getattr(self.config, 'postgres_url', 
                            "postgres://doc_user:changeme@postgres:5432/doc_ai")
            # Parse DSN for connection pool
            # Simple hardcoded fallback for now
            self.pg_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host="postgres",
                port=5432,
                database="doc_ai",
                user="doc_user",
                password=os.environ.get("POSTGRES_PASSWORD", "changeme")
            )
            logger.info("PostgreSQL pool initialized for long‑term memory")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            self.pg_pool = None

    # ------------------------------------------------------------------
    # Short‑term memory (Redis)
    # ------------------------------------------------------------------

    async def set_short_term(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store a value in Redis with optional TTL."""
        if not self.redis_client:
            logger.warning("Redis client not available, skipping short‑term set")
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            ttl = ttl_seconds or self.config.memory.short_term_ttl_seconds
            await self.redis_client.setex(key, ttl, value)
            logger.debug(f"Short‑term memory set: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set short‑term memory: {e}")
            return False

    async def get_short_term(self, key: str) -> Optional[Any]:
        """Retrieve a value from Redis."""
        if not self.redis_client:
            logger.warning("Redis client not available, skipping short‑term get")
            return None

        try:
            value = await self.redis_client.get(key)
            if value:
                # Try to parse as JSON, fallback to raw string
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get short‑term memory: {e}")
            return None

    async def delete_short_term(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self.redis_client:
            return False
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete short‑term memory: {e}")
            return False

    # ------------------------------------------------------------------
    # Long‑term memory (PostgreSQL)
    # ------------------------------------------------------------------

    def _get_pg_connection(self):
        """Get a connection from the PostgreSQL pool."""
        if not self.pg_pool:
            raise ConnectionError("PostgreSQL pool not initialized")
        return self.pg_pool.getconn()

    def _put_pg_connection(self, conn):
        """Return a connection to the PostgreSQL pool."""
        if self.pg_pool:
            self.pg_pool.putconn(conn)

    def store_long_term(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a value in PostgreSQL for long‑term persistence.
        """
        if not self.pg_pool:
            logger.error("PostgreSQL pool not available")
            return False

        conn = None
        try:
            conn = self._get_pg_connection()
            cursor = conn.cursor()

            # Ensure the memory table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doca_memory (
                    id SERIAL PRIMARY KEY,
                    key VARCHAR(255) NOT NULL,
                    namespace VARCHAR(100) NOT NULL,
                    value TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(key, namespace)
                )
            """)

            # Serialize value as JSON
            if isinstance(value, (dict, list)):
                value_json = json.dumps(value)
            else:
                value_json = str(value)

            metadata_json = json.dumps(metadata or {})

            # Upsert
            cursor.execute("""
                INSERT INTO doca_memory (key, namespace, value, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key, namespace) DO UPDATE
                SET value = EXCLUDED.value,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, namespace, value_json, metadata_json))

            conn.commit()
            logger.info(f"Long‑term memory stored: {namespace}/{key}")
            return True

        except Exception as e:
            logger.error(f"Failed to store long‑term memory: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._put_pg_connection(conn)

    def get_long_term(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Retrieve a value from PostgreSQL.
        """
        if not self.pg_pool:
            logger.error("PostgreSQL pool not available")
            return None

        conn = None
        try:
            conn = self._get_pg_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT value, metadata FROM doca_memory
                WHERE key = %s AND namespace = %s
            """, (key, namespace))

            row = cursor.fetchone()
            if not row:
                return None

            value_str = row[0]
            # Attempt to parse as JSON, fallback to raw string
            try:
                return json.loads(value_str)
            except:
                return value_str

        except Exception as e:
            logger.error(f"Failed to get long‑term memory: {e}")
            return None
        finally:
            if conn:
                self._put_pg_connection(conn)

    def delete_long_term(self, key: str, namespace: str = "default") -> bool:
        """Delete a record from PostgreSQL."""
        if not self.pg_pool:
            return False
        conn = None
        try:
            conn = self._get_pg_connection()
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM doca_memory WHERE key = %s AND namespace = %s
            """, (key, namespace))
            conn.commit()
            logger.info(f"Long‑term memory deleted: {namespace}/{key}")
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete long‑term memory: {e}")
            return False
        finally:
            if conn:
                self._put_pg_connection(conn)

    # ------------------------------------------------------------------
    # Vector retrieval (placeholder for Milvus)
    # ------------------------------------------------------------------

    def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the vector store.
        This is a placeholder; Milvus integration will be added later.
        """
        logger.warning("Vector search not yet implemented (Milvus integration pending)")
        return []

    # ------------------------------------------------------------------
    # Graph query (placeholder for Neo4j)
    # ------------------------------------------------------------------

    def query_graph(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the graph database.
        This is a placeholder; Neo4j integration will be added later.
        """
        logger.warning("Graph query not yet implemented (Neo4j integration pending)")
        return []

    # ------------------------------------------------------------------
    # Context aggregation
    # ------------------------------------------------------------------

    async def get_context(self, key: str, namespace: str = "default") -> Dict[str, Any]:
        """
        Aggregate context from all memory layers for a given key.
        Returns a dict with 'short_term', 'long_term', and optionally 'vector' results.
        """
        context = {
            "short_term": await self.get_short_term(key),
            "long_term": self.get_long_term(key, namespace),
            "vector": None,
            "graph": None,
        }
        return context

    async def store_context(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl_seconds: Optional[int] = None,
        persist_long_term: bool = True
    ) -> bool:
        """
        Store context across multiple memory layers.
        """
        success_short = await self.set_short_term(key, value, ttl_seconds)
        success_long = False
        if persist_long_term:
            success_long = self.store_long_term(key, value, namespace)

        return success_short or success_long