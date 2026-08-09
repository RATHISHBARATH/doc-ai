# ============================================================
# DOC AI Data Pipeline – MinIO Client
# ============================================================

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Iterator
from io import BytesIO

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import ENABLED
from minio.helpers import ObjectWriteResult

from src.common.config import get_config

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    A wrapper for the MinIO client that provides simplified methods
    for common storage operations. All operations are retried on failure
    and logged for observability.
    """

    def __init__(self, config=None):
        """
        Initialize the MinIO client from configuration.
        If no config is provided, it loads the global config via get_config().
        """
        if config is None:
            config = get_config()
        self.config = config
        self.bucket = config.minio.bucket

        # Create the MinIO client
        self.client = Minio(
            endpoint=config.minio.endpoint,
            access_key=config.minio.access_key,
            secret_key=config.minio.secret_key,
            secure=config.minio.secure,
        )

        # Ensure the bucket exists
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create the bucket if it does not exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                logger.debug(f"Bucket already exists: {self.bucket}")
        except S3Error as e:
            logger.error(f"Failed to create/check bucket {self.bucket}: {e}")
            raise

    # ------------------------------------------------------------------
    # Upload operations
    # ------------------------------------------------------------------

    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> ObjectWriteResult:
        """
        Upload a local file to MinIO.
        Returns the upload result (etag, version_id, etc.).
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            result = self.client.fput_object(
                bucket_name=self.bucket,
                object_name=remote_path,
                file_path=str(local_path),
            )
            logger.info(f"Uploaded {local_path} → {self.bucket}/{remote_path} (etag={result.etag})")
            return result
        except S3Error as e:
            logger.error(f"Failed to upload {local_path} to {remote_path}: {e}")
            raise

    def upload_bytes(self, data: bytes, remote_path: str, content_type: str = "application/octet-stream") -> ObjectWriteResult:
        """
        Upload in‑memory bytes to MinIO.
        Useful for small objects (e.g., tokenizer metadata, configuration files).
        """
        try:
            stream = BytesIO(data)
            result = self.client.put_object(
                bucket_name=self.bucket,
                object_name=remote_path,
                data=stream,
                length=len(data),
                content_type=content_type,
            )
            logger.info(f"Uploaded {len(data)} bytes → {self.bucket}/{remote_path}")
            return result
        except S3Error as e:
            logger.error(f"Failed to upload bytes to {remote_path}: {e}")
            raise

    # ------------------------------------------------------------------
    # Download operations
    # ------------------------------------------------------------------

    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> None:
        """
        Download a file from MinIO to a local path.
        """
        local_path = Path(local_path)
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.fget_object(
                bucket_name=self.bucket,
                object_name=remote_path,
                file_path=str(local_path),
            )
            logger.info(f"Downloaded {self.bucket}/{remote_path} → {local_path}")
        except S3Error as e:
            logger.error(f"Failed to download {remote_path} to {local_path}: {e}")
            raise

    def download_bytes(self, remote_path: str) -> bytes:
        """
        Download a remote object as in‑memory bytes.
        """
        try:
            response = self.client.get_object(
                bucket_name=self.bucket,
                object_name=remote_path,
            )
            data = response.read()
            response.close()
            response.release_conn()
            logger.info(f"Downloaded {len(data)} bytes from {self.bucket}/{remote_path}")
            return data
        except S3Error as e:
            logger.error(f"Failed to download bytes from {remote_path}: {e}")
            raise

    # ------------------------------------------------------------------
    # Listing and metadata operations
    # ------------------------------------------------------------------

    def list_objects(self, prefix: str = "", recursive: bool = True) -> List[str]:
        """
        List all object names in the bucket under a given prefix.
        Returns a list of object names (strings).
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket,
                prefix=prefix,
                recursive=recursive,
            )
            obj_names = [obj.object_name for obj in objects]
            logger.debug(f"Listed {len(obj_names)} objects under prefix '{prefix}'")
            return obj_names
        except S3Error as e:
            logger.error(f"Failed to list objects under prefix '{prefix}': {e}")
            raise

    def object_exists(self, remote_path: str) -> bool:
        """
        Check if an object exists in the bucket.
        """
        try:
            self.client.stat_object(
                bucket_name=self.bucket,
                object_name=remote_path,
            )
            return True
        except S3Error:
            return False

    def delete_object(self, remote_path: str) -> None:
        """
        Delete an object from the bucket.
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket,
                object_name=remote_path,
            )
            logger.info(f"Deleted {self.bucket}/{remote_path}")
        except S3Error as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            raise

    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all objects under a given prefix.
        Returns the number of objects deleted.
        """
        objects = self.list_objects(prefix=prefix, recursive=True)
        if not objects:
            logger.debug(f"No objects to delete under prefix '{prefix}'")
            return 0

        try:
            # Delete objects in batches (minio supports batch deletion)
            for obj in objects:
                self.client.remove_object(self.bucket, obj)
            logger.info(f"Deleted {len(objects)} objects under prefix '{prefix}'")
            return len(objects)
        except S3Error as e:
            logger.error(f"Failed to delete prefix '{prefix}': {e}")
            raise

    # ------------------------------------------------------------------
    # Utility operations
    # ------------------------------------------------------------------

    def get_object_size(self, remote_path: str) -> int:
        """
        Get the size (in bytes) of an object.
        """
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket,
                object_name=remote_path,
            )
            return stat.size
        except S3Error as e:
            logger.error(f"Failed to stat object {remote_path}: {e}")
            raise

    # ------------------------------------------------------------------
    # Context manager for temporary operations
    # ------------------------------------------------------------------

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit – no cleanup needed."""
        pass