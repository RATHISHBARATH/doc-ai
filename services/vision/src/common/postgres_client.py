# ============================================================
# DOC AI Vision Service – PostgreSQL Client
# ============================================================

import logging
import json
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union

import psycopg2
from psycopg2 import pool, extras, sql

from src.common.config import get_config

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    PostgreSQL client with connection pooling and simplified CRUD methods.
    All database operations are logged and include error handling.
    """

    def __init__(self, config=None):
        """
        Initialize the PostgreSQL client from configuration.
        If no config is provided, it loads the global config via get_config().
        """
        if config is None:
            config = get_config()
        self.config = config

        # Create the connection pool (credentials from env or defaults)
        self.pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.environ.get("VISION_POSTGRES_HOST", "postgres"),
            port=int(os.environ.get("VISION_POSTGRES_PORT", "5432")),
            database=os.environ.get("VISION_POSTGRES_DATABASE", "doc_ai"),
            user=os.environ.get("VISION_POSTGRES_USER", "doc_user"),
            password=os.environ.get("VISION_POSTGRES_PASSWORD", "changeme"),
        )
        logger.info(f"PostgreSQL connection pool created for Vision service")

        # Ensure the schema is set up
        self._ensure_schema()

    @contextmanager
    def get_connection(self):
        """
        Context manager that yields a connection from the pool.
        Automatically returns the connection to the pool on exit.
        """
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = False):
        """
        Context manager that yields a cursor from a pooled connection.
        If commit=True, the transaction is committed on exit (unless an exception occurs).
        """
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
        """
        Create the necessary tables if they do not exist.
        This ensures the database is ready for use.
        """
        with self.get_cursor(commit=True) as cursor:
            # Table: vision_processing_jobs (tracks video/image processing runs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vision_processing_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL UNIQUE,
                    source_url TEXT NOT NULL,
                    media_type VARCHAR(20) CHECK (media_type IN ('image', 'video')),
                    status VARCHAR(20) CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    metadata JSONB
                )
            """)

            # Table: scenes (stores video scene metadata and embeddings)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL REFERENCES vision_processing_jobs(job_id) ON DELETE CASCADE,
                    scene_index INTEGER NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    description TEXT,
                    embedding VECTOR(512),         -- Milvus will store the actual vector, but we keep metadata here
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)

            # Table: detections (stores object detection results)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL REFERENCES vision_processing_jobs(job_id) ON DELETE CASCADE,
                    frame_number INTEGER,
                    label VARCHAR(255) NOT NULL,
                    confidence FLOAT NOT NULL,
                    x FLOAT,
                    y FLOAT,
                    width FLOAT,
                    height FLOAT,
                    metadata JSONB
                )
            """)

            # Table: ocr_results (stores OCR text from images/video frames)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL REFERENCES vision_processing_jobs(job_id) ON DELETE CASCADE,
                    frame_number INTEGER,
                    text TEXT NOT NULL,
                    confidence FLOAT,
                    bounding_box JSONB,
                    metadata JSONB
                )
            """)

            # Table: faces (stores face detection/recognition results)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100) NOT NULL REFERENCES vision_processing_jobs(job_id) ON DELETE CASCADE,
                    frame_number INTEGER,
                    x FLOAT,
                    y FLOAT,
                    width FLOAT,
                    height FLOAT,
                    name VARCHAR(255),
                    confidence FLOAT,
                    landmarks JSONB,
                    metadata JSONB
                )
            """)

            logger.info("PostgreSQL schema initialized for Vision service (tables created if not existing)")

    # ------------------------------------------------------------------
    # Job metadata operations
    # ------------------------------------------------------------------

    def insert_job(self, job_id: str, source_url: str, media_type: str,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert a new processing job. Returns the job_id.
        """
        metadata_json = json.dumps(metadata) if metadata else '{}'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO vision_processing_jobs
                (job_id, source_url, media_type, status, metadata)
                VALUES (%s, %s, %s, 'pending', %s)
                RETURNING job_id
            """, (job_id, source_url, media_type, metadata_json))
            result = cursor.fetchone()
            job_id_out = result[0] if result else None
            logger.info(f"Inserted vision job: job_id={job_id_out}")
            return job_id_out

    def update_job_status(self, job_id: str, status: str,
                          error_message: Optional[str] = None) -> None:
        """
        Update the status of a processing job.
        """
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                UPDATE vision_processing_jobs
                SET status = %s,
                    error_message = %s
                WHERE job_id = %s
            """, (status, error_message, job_id))
            logger.info(f"Updated job status: job_id={job_id}, status={status}")

    def start_job(self, job_id: str) -> None:
        """
        Mark a job as started.
        """
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                UPDATE vision_processing_jobs
                SET status = 'processing',
                    started_at = CURRENT_TIMESTAMP
                WHERE job_id = %s
            """, (job_id,))
            logger.info(f"Started job: job_id={job_id}")

    def complete_job(self, job_id: str, error_message: Optional[str] = None) -> None:
        """
        Mark a job as completed or failed.
        """
        status = 'failed' if error_message else 'completed'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                UPDATE vision_processing_jobs
                SET status = %s,
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = %s
                WHERE job_id = %s
            """, (status, error_message, job_id))
            logger.info(f"Completed job: job_id={job_id}, status={status}")

    # ------------------------------------------------------------------
    # Scene operations
    # ------------------------------------------------------------------

    def insert_scene(self, job_id: str, scene_index: int, start_frame: int,
                     end_frame: int, description: Optional[str] = None,
                     confidence: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a scene record. Returns the generated ID.
        """
        metadata_json = json.dumps(metadata) if metadata else '{}'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO scenes
                (job_id, scene_index, start_frame, end_frame, description, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (job_id, scene_index, start_frame, end_frame, description, confidence, metadata_json))
            result = cursor.fetchone()
            scene_id = result[0] if result else None
            logger.debug(f"Inserted scene: job_id={job_id}, scene_index={scene_index}, id={scene_id}")
            return scene_id

    def get_scenes_by_job(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all scenes for a given job.
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, scene_index, start_frame, end_frame, description, confidence, created_at, metadata
                FROM scenes
                WHERE job_id = %s
                ORDER BY scene_index ASC
            """, (job_id,))
            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'scene_index': row[1],
                    'start_frame': row[2],
                    'end_frame': row[3],
                    'description': row[4],
                    'confidence': row[5],
                    'created_at': row[6],
                    'metadata': row[7],
                }
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Detection operations
    # ------------------------------------------------------------------

    def insert_detection(self, job_id: str, frame_number: int, label: str,
                         confidence: float, x: float, y: float,
                         width: float, height: float,
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a detection record. Returns the generated ID.
        """
        metadata_json = json.dumps(metadata) if metadata else '{}'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO detections
                (job_id, frame_number, label, confidence, x, y, width, height, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (job_id, frame_number, label, confidence, x, y, width, height, metadata_json))
            result = cursor.fetchone()
            detection_id = result[0] if result else None
            logger.debug(f"Inserted detection: job_id={job_id}, label={label}")
            return detection_id

    # ------------------------------------------------------------------
    # OCR results operations
    # ------------------------------------------------------------------

    def insert_ocr_result(self, job_id: str, frame_number: int, text: str,
                          confidence: Optional[float] = None,
                          bounding_box: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert an OCR result. Returns the generated ID.
        """
        metadata_json = json.dumps(metadata) if metadata else '{}'
        bbox_json = json.dumps(bounding_box) if bounding_box else '[]'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO ocr_results
                (job_id, frame_number, text, confidence, bounding_box, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (job_id, frame_number, text, confidence, bbox_json, metadata_json))
            result = cursor.fetchone()
            ocr_id = result[0] if result else None
            logger.debug(f"Inserted OCR result: job_id={job_id}")
            return ocr_id

    # ------------------------------------------------------------------
    # Face operations
    # ------------------------------------------------------------------

    def insert_face(self, job_id: str, frame_number: int, x: float, y: float,
                    width: float, height: float, name: Optional[str] = None,
                    confidence: Optional[float] = None,
                    landmarks: Optional[List[Dict[str, float]]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a face record. Returns the generated ID.
        """
        metadata_json = json.dumps(metadata) if metadata else '{}'
        landmarks_json = json.dumps(landmarks) if landmarks else '[]'
        with self.get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO faces
                (job_id, frame_number, x, y, width, height, name, confidence, landmarks, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (job_id, frame_number, x, y, width, height, name, confidence, landmarks_json, metadata_json))
            result = cursor.fetchone()
            face_id = result[0] if result else None
            logger.debug(f"Inserted face: job_id={job_id}")
            return face_id

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def execute_query(self, query: str, params: tuple = (),
                      commit: bool = False, fetch: bool = False) -> Any:
        """
        Execute an arbitrary SQL query.
        Useful for ad‑hoc queries and debugging.
        """
        with self.get_cursor(commit=commit) as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None

    # ------------------------------------------------------------------
    # Context manager for the client itself
    # ------------------------------------------------------------------

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit – close the connection pool."""
        self.pool.closeall()
        logger.info("PostgreSQL connection pool closed")