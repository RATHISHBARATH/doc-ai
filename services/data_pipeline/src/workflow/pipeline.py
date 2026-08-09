# ============================================================
# DOC AI Data Pipeline – Prefect Workflow (Fully Corrected)
# ============================================================

import logging
import os
from pathlib import Path

from prefect import flow, task

# Import configuration and clients
from src.common.config import get_config
from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient

# Import pipeline stages
from src.ingestion.download import download_data
from src.cleaning.clean_text import clean_data
from src.deduplication.dedup import deduplicate
from src.quality_filter.filter import quality_filter
from src.tokenizer_train.train import train_tokenizer
from src.dataset_prep.tokenize_dataset import tokenize_dataset
from src.dataset_prep.chunk import chunk_data
from src.dataset_prep.format_parquet import format_parquet

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Task Definitions
# ------------------------------------------------------------------

@task(retries=3, retry_delay_seconds=5)
def task_download_data(config_path: str) -> None:
    """Download raw data from configured sources."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        download_data(config, client)
        logger.info("Data download completed.")
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_clean_data(config_path: str) -> None:
    """Clean the raw data."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        clean_data(config, client)
        logger.info("Data cleaning completed.")
    except Exception as e:
        logger.error(f"Cleaning failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_deduplicate(config_path: str) -> None:
    """Deduplicate the cleaned data."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        deduplicate(config, client)
        logger.info("Deduplication completed.")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_quality_filter(config_path: str) -> None:
    """Apply quality filtering to the deduplicated data."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        quality_filter(config, client)
        logger.info("Quality filtering completed.")
    except Exception as e:
        logger.error(f"Quality filter failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_train_tokenizer(config_path: str) -> None:
    """Train the BPE tokenizer on the filtered data."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        db = PostgresClient(config)
        train_tokenizer(config, client, db)
        logger.info("Tokenizer training completed.")
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_tokenize_dataset(config_path: str) -> None:
    """Tokenize the filtered data using the trained tokenizer."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        db = PostgresClient(config)
        tokenize_dataset(config, client, db)
        logger.info("Dataset tokenization completed.")
    except Exception as e:
        logger.error(f"Tokenization failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_chunk_data(config_path: str) -> None:
    """Chunk the tokenized data into training examples."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        chunk_data(config, client)
        logger.info("Data chunking completed.")
    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        raise


@task(retries=3, retry_delay_seconds=5)
def task_format_parquet(config_path: str) -> None:
    """Format the chunked data into Parquet files."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        format_parquet(config, client)
        logger.info("Parquet formatting completed.")
    except Exception as e:
        logger.error(f"Parquet formatting failed: {e}", exc_info=True)
        raise


@task
def task_cleanup_intermediate(config_path: str) -> None:
    """Delete intermediate files to free up storage."""
    try:
        config = get_config(config_path)
        client = MinIOClient(config)
        prefixes = ["cleaned/", "deduped/", "filtered/", "tokenized/"]
        for prefix in prefixes:
            client.delete_prefix(prefix)
        logger.info("Intermediate files cleaned up.")
    except Exception as e:
        logger.warning(f"Cleanup failed (non‑critical): {e}", exc_info=True)


# ------------------------------------------------------------------
# Flow Definition
# ------------------------------------------------------------------

@flow(name="DOC AI Data Pipeline")
def data_pipeline(config_path: str = "configs/development.yaml") -> None:
    """
    The main Prefect flow that orchestrates the entire data pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # Validate that config_path is a string and the file exists
    if not isinstance(config_path, str):
        raise TypeError(f"config_path must be a string, got {type(config_path)}")
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Starting data pipeline with config: {config_path}")

    task_download_data(config_path)
    task_clean_data(config_path)
    task_deduplicate(config_path)
    task_quality_filter(config_path)
    task_train_tokenizer(config_path)
    task_tokenize_dataset(config_path)
    task_chunk_data(config_path)
    task_format_parquet(config_path)

    task_cleanup_intermediate(config_path)

    logger.info("Data pipeline completed successfully.")


# ------------------------------------------------------------------
# Entry point for running the flow directly
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Determine config path: from command line, environment, or default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.environ.get("DOC_CONFIG_PATH", "configs/development.yaml")
    data_pipeline(config_path)