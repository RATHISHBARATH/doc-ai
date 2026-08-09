# ============================================================
# DOC AI Data Pipeline – Dataset Tokenization Module
# ============================================================

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient
from src.common.config import Config

logger = logging.getLogger(__name__)


def tokenize_dataset(config: Config, client: MinIOClient, db: PostgresClient) -> None:
    """
    Main entry point: load tokenizer, tokenize filtered data, and upload to MinIO.
    """
    # 1. Determine which tokenizer version to use from config or DB
    # For simplicity, we'll load the latest tokenizer version from MinIO.
    # In production, we'd retrieve the version from the database.
    tokenizer_version = f"v{config.tokenizer.vocab_size}_{config.tokenizer.min_frequency}"
    tokenizer_path = f"tokenizer/{tokenizer_version}/tokenizer.json"
    
    # 2. Download the tokenizer artifact from MinIO
    local_tokenizer_dir = Path("/tmp/tokenizer")
    local_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    local_tokenizer_path = local_tokenizer_dir / "tokenizer.json"
    client.download_file(tokenizer_path, local_tokenizer_path)
    logger.info(f"Downloaded tokenizer from {tokenizer_path}")

    # 3. Load the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(local_tokenizer_path),
        padding_side="right",
        truncation_side="right",
    )
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    logger.info(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}")

    # 4. List filtered objects in MinIO
    filtered_objects = client.list_objects(prefix="filtered/", recursive=True)
    if not filtered_objects:
        logger.warning("No filtered data found. Skipping tokenization.")
        return

    # 5. Download filtered files and load into a Dataset
    local_files = []
    for obj in filtered_objects:
        local_path = f"/tmp/filtered_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)
        local_files.append(local_path)

    dataset = load_dataset(
        "parquet",
        data_files=local_files,
        split="train",
    )
    logger.info(f"Loaded {len(dataset)} documents for tokenization")

    # 6. Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=dataset.column_names,
    )
    logger.info(f"Tokenized {len(tokenized_dataset)} documents")

    # 7. Save tokenized data to a local Parquet file
    local_output = Path("/tmp/tokenized_output.parquet")
    tokenized_dataset.to_parquet(str(local_output))
    logger.info(f"Saved tokenized dataset to {local_output}")

    # 8. Upload to MinIO
    remote_path = f"tokenized/{tokenizer_version}/dataset.parquet"
    client.upload_file(str(local_output), remote_path)
    logger.info(f"Uploaded tokenized dataset to {remote_path}")

    # 9. Register dataset metadata in PostgreSQL
    # (We will do this in a separate step; for now we just upload.)

    # 10. Clean up temporary files
    for file in local_files:
        os.remove(file)
    os.remove(local_output)
    os.remove(local_tokenizer_path)
    os.rmdir(local_tokenizer_dir)
    logger.info("Cleaned up temporary files")


def tokenize_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerFast,
    config: Config,
) -> Dict[str, List]:
    """
    Tokenize a batch of documents.
    Returns a dict with 'input_ids' and 'attention_mask'.
    """
    texts = examples.get('cleaned_text', [])
    if not texts:
        return {'input_ids': [], 'attention_mask': []}

    # Tokenize with padding and truncation
    max_length = config.dataset_prep.max_length
    outputs = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    # Convert to Python lists for storage
    return {
        'input_ids': outputs['input_ids'].tolist(),
        'attention_mask': outputs['attention_mask'].tolist(),
    }