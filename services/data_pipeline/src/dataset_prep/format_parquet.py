# ============================================================
# DOC AI Data Pipeline – Parquet Formatting Module
# ============================================================

import logging
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.minio_client import MinIOClient
from src.common.config import Config

logger = logging.getLogger(__name__)


def format_parquet(config: Config, client: MinIOClient) -> None:
    """
    Main entry point: read chunked data, format as Parquet with proper schema,
    add metadata, and upload final dataset to MinIO.
    """
    # List chunked objects in MinIO
    chunked_objects = client.list_objects(prefix="chunked/", recursive=True)
    if not chunked_objects:
        logger.warning("No chunked data found. Skipping Parquet formatting.")
        return

    # We'll process the chunked file(s) and produce a single final dataset.
    # For simplicity, we assume there's one chunked file (or we merge them).
    final_dfs = []
    for obj in chunked_objects:
        local_path = f"/tmp/chunked_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)
        df = pd.read_parquet(local_path)
        final_dfs.append(df)
        os.remove(local_path)

    if not final_dfs:
        logger.warning("No data loaded from chunked files.")
        return

    # Combine all DataFrames
    combined_df = pd.concat(final_dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} rows from chunked data")

    # Ensure required columns exist and rename if necessary
    # Expected columns: input_ids, attention_mask, labels (for causal LM, labels = input_ids)
    if 'input_ids' not in combined_df.columns:
        raise ValueError("Missing 'input_ids' column")
    if 'attention_mask' not in combined_df.columns:
        raise ValueError("Missing 'attention_mask' column")

    # Add labels column (for causal language modeling, labels equal input_ids)
    combined_df['labels'] = combined_df['input_ids']

    # Reorder columns to match training pipeline expectations
    combined_df = combined_df[['input_ids', 'attention_mask', 'labels']]

    # Add metadata
    metadata = {
        'dataset_version': 'v1.0',
        'tokenizer_version': f"v{config.tokenizer.vocab_size}_{config.tokenizer.min_frequency}",
        'max_length': config.dataset_prep.max_length,
        'stride': config.dataset_prep.stride,
        'created_at': datetime.utcnow().isoformat(),
        'num_documents': len(combined_df),
    }

    # Write to a single Parquet file with metadata
    local_final_path = Path("/tmp/final_dataset.parquet")
    table = pa.Table.from_pandas(combined_df)
    table = table.replace_schema_metadata({
        'DOC_AI_METADATA': json.dumps(metadata)
    })
    pq.write_table(table, local_final_path, compression='snappy')
    logger.info(f"Wrote final dataset to {local_final_path}")

    # Upload to MinIO
    remote_path = f"final/{metadata['dataset_version']}/dataset.parquet"
    client.upload_file(str(local_final_path), remote_path)
    logger.info(f"Uploaded final dataset to {remote_path}")

    # Clean up
    os.remove(local_final_path)
    logger.info("Cleaned up temporary files")