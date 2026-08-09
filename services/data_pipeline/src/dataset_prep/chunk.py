# ============================================================
# DOC AI Data Pipeline – Dataset Chunking Module (Corrected)
# ============================================================

import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from src.common.minio_client import MinIOClient
from src.common.config import Config

logger = logging.getLogger(__name__)


def chunk_data(config: Config, client: MinIOClient) -> None:
    """
    Main entry point: download tokenized data, chunk into training examples,
    and upload chunked data.
    """
    tokenized_objects = client.list_objects(prefix="tokenized/", recursive=True)
    if not tokenized_objects:
        logger.warning("No tokenized data found. Skipping chunking.")
        return

    for obj in tokenized_objects:
        local_path = f"/tmp/tokenized_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)

        # Chunk the file
        chunked_df = chunk_file(local_path, config)

        # Upload chunked data
        chunked_remote = obj.replace("tokenized/", "chunked/")
        client.upload_file(local_path, chunked_remote)

        # Clean up
        os.remove(local_path)
    logger.info("Chunking completed.")


def chunk_file(local_path: str, config: Config) -> pd.DataFrame:
    """
    Read a Parquet file of tokenized documents and split them into
    overlapping chunks of a fixed max_length.
    Returns a DataFrame of chunked examples.
    """
    df = pd.read_parquet(local_path)
    if df.empty:
        logger.warning(f"File {local_path} is empty – skipping chunking")
        return df

    # Expecting columns: 'input_ids' (list of ints), 'attention_mask' (list of ints)
    if 'input_ids' not in df.columns or 'attention_mask' not in df.columns:
        raise ValueError("Tokenized data must contain 'input_ids' and 'attention_mask' columns")

    max_length = config.dataset_prep.max_length
    stride = config.dataset_prep.stride

    chunked_rows = []
    for _, row in df.iterrows():
        ids = row['input_ids']
        mask = row['attention_mask']
        # Chunk the sequences
        for start in range(0, len(ids), stride):
            end = start + max_length
            chunk_ids = ids[start:end]
            chunk_mask = mask[start:end]
            # Pad if necessary (to max_length)
            if len(chunk_ids) < max_length:
                pad_len = max_length - len(chunk_ids)
                chunk_ids = chunk_ids + [0] * pad_len
                chunk_mask = chunk_mask + [0] * pad_len
            chunked_rows.append({
                'input_ids': chunk_ids,
                'attention_mask': chunk_mask,
            })

    chunked_df = pd.DataFrame(chunked_rows)
    logger.info(f"Chunked {len(df)} sequences into {len(chunked_df)} examples")
    return chunked_df