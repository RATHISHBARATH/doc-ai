# ============================================================
# DOC AI Data Pipeline – Deduplication Entry Point (Path Fix)
# ============================================================

import logging
import os
import tempfile
from pathlib import Path

from src.common.minio_client import MinIOClient
from src.common.config import Config
from src.deduplication.minhash_lsh import deduplicate_file as minhash_dedup

logger = logging.getLogger(__name__)


def deduplicate(config: Config, client: MinIOClient) -> None:
    """
    Main entry point: list cleaned data, deduplicate, and upload.
    """
    try:
        cleaned_objects = list(client.list_objects(prefix="cleaned/", recursive=True))
    except Exception as e:
        logger.error(f"Failed to list cleaned objects from MinIO: {e}", exc_info=True)
        return

    if not cleaned_objects:
        logger.warning("No cleaned data found in MinIO. Skipping deduplication.")
        return

    logger.info(f"Found {len(cleaned_objects)} cleaned objects to process.")

    for obj in cleaned_objects:
        local_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"cleaned_{obj.replace('/', '_')}_"
            ) as tmp_file:
                local_path = tmp_file.name

            logger.info(f"Downloading {obj} to {local_path}")
            client.download_file(obj, local_path)

            # Convert to Path before passing to minhash_dedup
            minhash_dedup(Path(local_path), config, client)

            deduped_remote = obj.replace("cleaned/", "deduped/")
            client.upload_file(local_path, deduped_remote)
            logger.info(f"Uploaded deduplicated data to {deduped_remote}")

        except Exception as e:
            logger.error(f"Failed to process object {obj}: {e}", exc_info=True)
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logger.debug(f"Removed temporary file {local_path}")
                except OSError as rm_err:
                    logger.warning(f"Could not remove temporary file {local_path}: {rm_err}")