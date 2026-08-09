# ============================================================
# DOC AI Data Pipeline – Tokenizer Training Module (Final Fix)
# ============================================================

import logging
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder
from tokenizers.normalizers import NFKC

from datasets import Dataset, load_dataset

from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient
from src.common.config import Config

logger = logging.getLogger(__name__)


def train_tokenizer(config: Config, client: MinIOClient, db: PostgresClient) -> None:
    """
    Main entry point: sample filtered data, train tokenizer, upload artifacts, and register metadata.
    """
    # 1. List filtered objects in MinIO
    filtered_objects = client.list_objects(prefix="filtered/", recursive=True)
    if not filtered_objects:
        logger.warning("No filtered data found. Skipping tokenizer training.")
        return

    # 2. Download filtered files and load into a Dataset
    local_files = []
    for obj in filtered_objects:
        local_path = f"/tmp/filtered_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)
        local_files.append(local_path)

    # 3. Load the dataset from Parquet files
    dataset = load_dataset(
        "parquet",
        data_files=local_files,
        split="train",
    )
    logger.info(f"Loaded {len(dataset)} documents from filtered data")

    # 4. Sample a subset for tokenizer training
    sample_size = config.tokenizer.sample_size  # in bytes (approx)
    sample_docs = sample_dataset(dataset, sample_size)
    logger.info(f"Sampled {len(sample_docs)} documents for tokenizer training")

    # 5. Train the tokenizer
    tokenizer = train_bpe(
        sample_docs,
        vocab_size=config.tokenizer.vocab_size,
        special_tokens=config.tokenizer.special_tokens,
        min_frequency=config.tokenizer.min_frequency,
    )

    # 6. Save tokenizer artifacts temporarily
    local_dir = Path("/tmp/tokenizer")
    local_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = local_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Saved tokenizer.json to {tokenizer_path}")

    # 7. Upload to MinIO (only tokenizer.json is needed)
    # Include timestamp to make the remote path unique
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version = f"v{config.tokenizer.vocab_size}_{config.tokenizer.min_frequency}_{timestamp}"
    remote_base = f"tokenizer/{version}/"
    client.upload_file(str(tokenizer_path), remote_base + "tokenizer.json")
    logger.info(f"Uploaded tokenizer artifacts to {remote_base}")

    # 8. Register metadata in PostgreSQL
    tokenizer_id = db.insert_tokenizer(
        version=version,
        vocab_size=config.tokenizer.vocab_size,
        special_tokens=config.tokenizer.special_tokens,
        min_frequency=config.tokenizer.min_frequency,
        trained_on="filtered_data_sample",
        minio_path=remote_base,
        metadata={
            "sample_size_bytes": sample_size,
            "num_documents": len(sample_docs),
        }
    )
    logger.info(f"Registered tokenizer in PostgreSQL with ID {tokenizer_id}")

    # 9. Clean up temporary files
    for file in local_files:
        os.remove(file)
    if local_dir.exists():
        for file in local_dir.glob("*"):
            file.unlink()
        local_dir.rmdir()
    logger.info("Cleaned up temporary files")


def sample_dataset(dataset: Dataset, target_size_bytes: int) -> List[str]:
    sample_texts = []
    total_size = 0
    for doc in dataset:
        text = doc.get('cleaned_text', '')
        if not text:
            continue
        sample_texts.append(text)
        total_size += len(text.encode('utf-8'))
        if total_size >= target_size_bytes:
            break
    return sample_texts


def train_bpe(
    texts: List[str],
    vocab_size: int = 100_000,
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2,
) -> Tokenizer:
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    logger.info(f"BPE tokenizer trained with vocab_size={vocab_size}")
    return tokenizer