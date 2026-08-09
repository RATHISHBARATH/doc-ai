# ============================================================
# DOC AI Data Pipeline – Ingestion Download Module
# ============================================================

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from tqdm import tqdm
from minio.error import S3Error

from src.common.minio_client import MinIOClient
from src.common.config import Config, DataSourceConfig

logger = logging.getLogger(__name__)


def download_data(config: Config, client: MinIOClient) -> None:
    """
    Download raw data from all configured sources and upload to MinIO.
    """
    for source in config.data_sources:
        logger.info(f"Downloading source: {source.name} from {source.url}")
        download_single_source(source, client, config)


def download_single_source(source: DataSourceConfig, client: MinIOClient, config: Config) -> None:
    """
    Download a single data source, upload to MinIO, and track processed documents.
    """
    # Generate a local temporary file path
    local_path = Path(config.data_root) / "raw" / f"{source.name}.{source.format}"
    if source.compression:
        local_path = local_path.with_suffix(f".{source.compression}")

    # Download to temporary file
    try:
        downloaded_path = download_with_retries(
            url=source.url,
            local_path=local_path,
            max_retries=config.workflow.max_retries,
            retry_delay=config.workflow.retry_delay_seconds,
        )
    except Exception as e:
        logger.error(f"Failed to download {source.name} from {source.url}: {e}")
        return

    # Upload to MinIO
    remote_path = f"raw/{source.name}/{local_path.name}"
    try:
        client.upload_file(downloaded_path, remote_path)
        logger.info(f"Uploaded {source.name} to {remote_path}")
    except S3Error as e:
        logger.error(f"Failed to upload {source.name} to MinIO: {e}")
        return

    # Clean up local file to save disk space
    try:
        downloaded_path.unlink()
        logger.debug(f"Removed local file: {downloaded_path}")
    except Exception as e:
        logger.warning(f"Could not remove local file {downloaded_path}: {e}")

    # If max_documents is set, we could truncate the file here
    # For now, we rely on the pipeline stages to respect the limit


def download_with_retries(
    url: str,
    local_path: Path,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Path:
    """
    Download a file from a URL with retries and progress bar.
    Returns the path to the downloaded file.
    """
    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries + 1):
        try:
            return _download_file(url, local_path)
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning(f"Download attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise RuntimeError(f"Failed to download {url} after {max_retries} retries")


def _download_file(url: str, local_path: Path) -> Path:
    """
    Actually perform the download with streaming and progress bar.
    """
    # Stream the download
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192  # 8 KB chunks

    # Write to file with progress bar
    with open(local_path, 'wb') as f:
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=f"Downloading {local_path.name}",
            ncols=80,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Downloaded {local_path.name} ({total_size} bytes)")
    return local_path


def truncate_file_to_max_documents(file_path: Path, max_docs: int, format: str) -> Path:
    """
    Truncate a file to keep only the first `max_docs` documents.
    This is a placeholder – actual implementation depends on file format.
    For JSONL files, we can read line by line and write the first N lines.
    For Parquet, we can use pandas/pyarrow.
    """
    # Implementation will be added later when needed.
    # For now, we just return the original file.
    logger.warning("truncate_file_to_max_documents is not yet implemented")
    return file_path