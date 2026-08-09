# ============================================================
# DOC AI Data Pipeline – Quality Filtering Entry Point
# ============================================================

import logging
import os
from pathlib import Path

from src.common.minio_client import MinIOClient
from src.common.config import Config
from src.quality_filter.quality_score import filter_file

logger = logging.getLogger(__name__)


def quality_filter(config: Config, client: MinIOClient) -> None:
    """
    Main entry point: list deduplicated data, apply quality filtering, and upload.
    """
    # List deduplicated objects in MinIO
    deduped_objects = client.list_objects(prefix="deduped/", recursive=True)
    if not deduped_objects:
        logger.warning("No deduplicated data found. Skipping quality filtering.")
        return

    for obj in deduped_objects:
        # Download the deduplicated file
        local_path = f"/tmp/deduped_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)

        # Apply quality filtering using the scoring module
        # filter_file will read the file, score each document, and write a new file
        # in the same location with only high‑quality documents.
        filter_file(local_path, config)

        # Upload the filtered data to MinIO
        filtered_remote = obj.replace("deduped/", "filtered/")
        client.upload_file(local_path, filtered_remote)

        # Clean up local temporary file
        os.remove(local_path)
        logger.info(f"Uploaded filtered data to {filtered_remote}")