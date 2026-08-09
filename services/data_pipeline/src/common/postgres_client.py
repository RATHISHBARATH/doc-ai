# ============================================================
# DOC AI Data Pipeline – PostgreSQL Client (Corrected)
# ============================================================

import logging
import json
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union

import psycopg2
from psycopg2 import pool

from src.common.config import get_config

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    PostgreSQL client with connection pooling and simplified CRUD methods.
    All database operations are logged and include error handling.
    """

    def __init__(self, config=None):
        if config is None:
            config = get_config()
        self.config = config
        self.pg_config = config.postgres

        self.pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=self.pg_config.host,
            port=self.pg_config.port,
            database=self.pg_config.database,
            user=self.pg_config.user,
            password=self.pg_config.password,
        )
        logger.info(f"PostgreSQL connection pool created (host={self.pg_config.host})")

        self._ensure_schema()

    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = False):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self.get_cursor(commit=True) as cursor:
            # Table: tokenizers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tokenizers (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    vocab_size INTEGER NOT NULL,
                    special_tokens JSONB,
                    min_frequency INTEGER,
                    trained_on VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    minio_path TEXT NOT NULL,
                    metadata JSONB
                )
            """)

            # Table: datasets
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    tokenizer_id INTEGER REFERENCES tokenizers(id),
                    source VARCHAR(255),
                    cleaning_params JSONB,
                    dedup_params JSONB,
                    quality_filter_params JSONB,
                    chunk_size INTEGER,
                    stride INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    minio_path TEXT NOT NULL,
                    metadata JSONB
                )
            """)

            # Table: pipeline_runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(100) NOT NULL UNIQUE,
                    dataset_id INTEGER REFERENCES datasets(id),
                    tokenizer_id INTEGER REFERENCES tokenizers(id),
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status VARCHAR(20) CHECK (status IN ('running', 'completed', 'failed')),
                    error_message TEXT,
                    processing_time_seconds FLOAT,
                    metadata JSONB
                )
            """)

            # Table: documents
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(255),
                    document_id VARCHAR(255),
                    hash VARCHAR(64),
                    status VARCHAR(20) CHECK (status IN ('pending', 'processed', 'failed')),
                    processed_at TIMESTAMP,
                    metadata JSONB,
                    UNIQUE(source, document_id)
                )
            """)

            logger.info("PostgreSQL schema initialized (tables created if not existing)")

    # ------------------------------------------------------------------
    # Tokenizer metadata operations
    # ------------------------------------------------------------------

    def insert_tokenizer(self, version: str, vocab_size: int, special_tokens: List[str],
                         min_frequency: int, trained_on: str, minio_path: str,
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a new tokenizer record. Returns the generated ID.
        """
        # JSON‑serialize any structured fields
        special_tokens_json = json.dumps(special_tokens) if special_tokens else '[]'
        metadata_json = json.dumps(metadata) if metadata else '{}'

        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO tokenizers
                (version, vocab_size, special_tokens, min_frequency, trained_on, minio_path, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                version, vocab_size, special_tokens_json, min_frequency,
                trained_on, minio_path, metadata_json
            ))
            result = cursor.fetchone()
            tokenizer_id = result[0] if result else None
            logger.info(f"Inserted tokenizer: version={version}, id={tokenizer_id}")
            return tokenizer_id

    def get_tokenizer_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, version, vocab_size, special_tokens, min_frequency,
                       trained_on, created_at, minio_path, metadata
                FROM tokenizers
                WHERE version = %s
            """, (version,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                'id': row[0],
                'version': row[1],
                'vocab_size': row[2],
                'special_tokens': row[3],
                'min_frequency': row[4],
                'trained_on': row[5],
                'created_at': row[6],
                'minio_path': row[7],
                'metadata': row[8],
            }

    # ------------------------------------------------------------------
    # Dataset metadata operations
    # ------------------------------------------------------------------

    def insert_dataset(self, version: str, tokenizer_id: int, source: str,
                       cleaning_params: Dict[str, Any], dedup_params: Dict[str, Any],
                       quality_filter_params: Dict[str, Any], chunk_size: int,
                       stride: int, minio_path: str,
                       metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a new dataset record. Returns the generated ID.
        """
        cleaning_params_json = json.dumps(cleaning_params) if cleaning_params else '{}'
        dedup_params_json = json.dumps(dedup_params) if dedup_params else '{}'
        quality_filter_params_json = json.dumps(quality_filter_params) if quality_filter_params else '{}'
        metadata_json = json.dumps(metadata) if metadata else '{}'

        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO datasets
                (version, tokenizer_id, source, cleaning_params, dedup_params,
                 quality_filter_params, chunk_size, stride, minio_path, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                version, tokenizer_id, source, cleaning_params_json, dedup_params_json,
                quality_filter_params_json, chunk_size, stride, minio_path, metadata_json
            ))
            result = cursor.fetchone()
            dataset_id = result[0] if result else None
            logger.info(f"Inserted dataset: version={version}, id={dataset_id}")
            return dataset_id

    def get_dataset_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT d.id, d.version, d.tokenizer_id, d.source,
                       d.cleaning_params, d.dedup_params, d.quality_filter_params,
                       d.chunk_size, d.stride, d.created_at, d.minio_path, d.metadata,
                       t.version as tokenizer_version
                FROM datasets d
                LEFT JOIN tokenizers t ON d.tokenizer_id = t.id
                WHERE d.version = %s
            """, (version,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                'id': row[0],
                'version': row[1],
                'tokenizer_id': row[2],
                'source': row[3],
                'cleaning_params': row[4],
                'dedup_params': row[5],
                'quality_filter_params': row[6],
                'chunk_size': row[7],
                'stride': row[8],
                'created_at': row[9],
                'minio_path': row[10],
                'metadata': row[11],
                'tokenizer_version': row[12],
            }

    # ------------------------------------------------------------------
    # Pipeline run metadata operations
    # ------------------------------------------------------------------

    def start_pipeline_run(self, run_id: str, dataset_id: int,
                           tokenizer_id: int, metadata: Optional[Dict[str, Any]] = None) -> int:
        metadata_json = json.dumps(metadata) if metadata else '{}'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO pipeline_runs
                (run_id, dataset_id, tokenizer_id, status, metadata)
                VALUES (%s, %s, %s, 'running', %s)
                RETURNING id
            """, (run_id, dataset_id, tokenizer_id, metadata_json))
            result = cursor.fetchone()
            run_id_db = result[0] if result else None
            logger.info(f"Started pipeline run: run_id={run_id}, id={run_id_db}")
            return run_id_db

    def finish_pipeline_run(self, run_id: str, status: str,
                            error_message: Optional[str] = None) -> None:
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                UPDATE pipeline_runs
                SET end_time = CURRENT_TIMESTAMP,
                    status = %s,
                    error_message = %s,
                    processing_time_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time))
                WHERE run_id = %s
            """, (status, error_message, run_id))
            logger.info(f"Finished pipeline run: run_id={run_id}, status={status}")

    # ------------------------------------------------------------------
    # Document tracking operations
    # ------------------------------------------------------------------

    def mark_document_processed(self, source: str, document_id: str,
                                hash: str, status: str = 'processed',
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        metadata_json = json.dumps(metadata) if metadata else '{}'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO documents (source, document_id, hash, status, processed_at, metadata)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s)
                ON CONFLICT (source, document_id) DO UPDATE
                SET hash = EXCLUDED.hash,
                    status = EXCLUDED.status,
                    processed_at = EXCLUDED.processed_at,
                    metadata = EXCLUDED.metadata
            """, (source, document_id, hash, status, metadata_json))
            logger.debug(f"Marked document: source={source}, doc_id={document_id}, status={status}")

    def get_document_status(self, source: str, document_id: str) -> Optional[str]:
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT status FROM documents
                WHERE source = %s AND document_id = %s
            """, (source, document_id))
            row = cursor.fetchone()
            return row[0] if row else None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def execute_query(self, query: str, params: tuple = (),
                      commit: bool = False, fetch: bool = False) -> Any:
        with self.get_cursor(commit=commit) as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None

    # ------------------------------------------------------------------
    # Context manager for the client itself
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.closeall()
        logger.info("PostgreSQL connection pool closed")