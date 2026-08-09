# ============================================================
# DOC AI Trainer – Data Loader (Return Tensors)
# ============================================================

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List

import torch
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from src.config import TrainerConfig
from src.common.minio_client import MinIOClient

logger = logging.getLogger(__name__)


class DOCDataset(IterableDataset):
    """
    An IterableDataset that loads documents from MinIO.
    If the dataset contains tokenized columns (`input_ids`, `attention_mask`, `labels`),
    they are yielded directly. Otherwise, it attempts to find a text column and tokenizes.
    All outputs are converted to PyTorch tensors.
    """

    def __init__(
        self,
        config: TrainerConfig,
        tokenizer: PreTrainedTokenizerBase,
        client: MinIOClient,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.client = client
        self.split = split
        self.max_length = config.data.max_seq_length

        # Download dataset from MinIO to a temporary file
        self._download_dataset()

        # Load as Hugging Face Dataset
        self.dataset = load_dataset(
            "parquet",
            data_files=str(self.local_dataset_path),
            split=self.split,
        )
        logger.info(f"Loaded {len(self.dataset)} examples from {self.local_dataset_path}")

    def _download_dataset(self) -> None:
        """Download the dataset from MinIO to a temporary local file."""
        remote_path = self.config.data.dataset_path
        if not remote_path:
            raise ValueError("No dataset path configured in data.dataset_path")

        # Resolve the dataset path
        resolved_path = _resolve_dataset_path(self.client, remote_path)
        if not resolved_path:
            raise FileNotFoundError(f"No dataset found matching {remote_path}")

        # Create a temporary file
        self.local_dataset_path = Path(tempfile.NamedTemporaryFile(
            prefix="doc_dataset_",
            suffix=".parquet",
            delete=False,
        ).name)

        # Download from MinIO
        logger.info(f"Downloading dataset {resolved_path} from MinIO to {self.local_dataset_path}")
        self.client.download_file(resolved_path, self.local_dataset_path)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset and yield tensors."""
        if len(self.dataset) == 0:
            logger.error("Dataset is empty.")
            return

        # Check first sample for tokenized columns
        sample = self.dataset[0]
        if "input_ids" in sample and "attention_mask" in sample:
            logger.info("Dataset already contains tokenized columns. Yielding them directly as tensors.")
            for example in self.dataset:
                # Convert lists to tensors (assume they are lists of ints)
                input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
                # Optionally include labels if present
                if "labels" in example:
                    labels = torch.tensor(example["labels"], dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                else:
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
            return

        # Fallback: detect text column and tokenize
        candidate_columns = ["cleaned_text", "text", "content", "body"]
        text_column = None
        for col in candidate_columns:
            if col in sample:
                text_column = col
                break
        if text_column is None:
            logger.error(f"Dataset does not contain any of the expected text columns: {candidate_columns}")
            logger.error(f"Available columns: {list(sample.keys())}")
            raise ValueError("No valid text column found in dataset.")

        logger.info(f"Using column '{text_column}' for text.")
        count = 0
        for example in self.dataset:
            text = example.get(text_column)
            if not text or not isinstance(text, str) or text.strip() == "":
                logger.debug(f"Skipping empty or missing '{text_column}': {example}")
                continue

            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,  # Return lists
            )

            if not tokenized["input_ids"]:
                logger.debug(f"Skipping example with no tokens: {text[:50]}...")
                continue

            count += 1
            # Convert lists to tensors
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        if count == 0:
            logger.error("No valid examples found after tokenization! Check your dataset.")
        else:
            logger.info(f"Yielded {count} valid tokenized examples.")

    def __len__(self) -> int:
        return len(self.dataset)


def _resolve_wildcard_path(client: MinIOClient, pattern: str) -> Optional[str]:
    if "*" not in pattern:
        return pattern

    parts = pattern.split("*")
    if len(parts) != 2:
        raise ValueError(f"Wildcard pattern must contain exactly one '*': {pattern}")
    prefix = parts[0]
    suffix = parts[1]

    objects = client.list_objects(prefix=prefix, recursive=True)
    matches = [obj for obj in objects if obj.startswith(prefix) and obj.endswith(suffix)]

    if not matches:
        logger.error(f"No objects found matching pattern {pattern}")
        return None

    matches.sort(reverse=True)
    selected = matches[0]
    logger.info(f"Resolved wildcard '{pattern}' to '{selected}'")
    return selected


def _resolve_dataset_path(client: MinIOClient, path: str) -> Optional[str]:
    if "*" in path:
        return _resolve_wildcard_path(client, path)

    if client.object_exists(path):
        return path

    prefix = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
    objects = client.list_objects(prefix=prefix, recursive=True)
    parquet_files = [obj for obj in objects if obj.endswith(".parquet")]
    if not parquet_files:
        logger.error(f"No Parquet files found under prefix '{prefix}'")
        return None

    parquet_files.sort(reverse=True)
    selected = parquet_files[0]
    logger.info(f"Resolved dataset path '{path}' to '{selected}'")
    return selected


def create_dataloader(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    client: MinIOClient,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    if batch_size is None:
        batch_size = config.training.per_device_train_batch_size

    dataset = DOCDataset(config, tokenizer, client, split=split)

    if shuffle and isinstance(dataset.dataset, Dataset):
        dataset.dataset = dataset.dataset.shuffle(seed=42)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def load_tokenizer_from_minio(
    config: TrainerConfig,
    client: MinIOClient,
) -> PreTrainedTokenizerBase:
    remote_path = config.data.tokenizer_path
    if not remote_path:
        raise ValueError("No tokenizer path configured in data.tokenizer_path")

    resolved_path = _resolve_wildcard_path(client, remote_path)
    if not resolved_path:
        raise FileNotFoundError(f"No tokenizer found matching {remote_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = Path(tmpdir)
        tokenizer_json_path = local_dir / "tokenizer.json"

        logger.info(f"Downloading tokenizer from {resolved_path} to {tokenizer_json_path}")
        client.download_file(resolved_path, tokenizer_json_path)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(local_dir),
            local_files_only=True,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}")
        return tokenizer