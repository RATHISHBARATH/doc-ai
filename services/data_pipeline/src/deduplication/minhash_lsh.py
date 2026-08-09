# ============================================================
# DOC AI Data Pipeline – MinHash LSH Deduplication (Corrected)
# ============================================================

import logging
import pandas as pd
from pathlib import Path
from typing import List, Set

from datasketch import MinHash, MinHashLSH

from src.common.minio_client import MinIOClient
from src.common.config import Config

logger = logging.getLogger(__name__)


def deduplicate_file(
    local_path: Path,
    config: Config,
    client: MinIOClient,
) -> None:
    """
    Read a file (Parquet or JSONL), deduplicate documents using MinHash LSH,
    and overwrite the file with the deduplicated DataFrame.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Input file {local_path} does not exist")

    # Attempt to read as Parquet first, fallback to JSONL
    try:
        df = pd.read_parquet(local_path)
    except Exception as e:
        logger.warning(f"Failed to read as Parquet ({e}), trying JSONL fallback")
        try:
            df = pd.read_json(local_path, lines=True)
        except Exception as e2:
            raise RuntimeError(f"Could not read file as Parquet or JSONL: {e2}")

    # Ensure we have a 'cleaned_text' column
    if 'cleaned_text' not in df.columns:
        raise ValueError("Input file must contain 'cleaned_text' column")

    if df.empty:
        logger.warning(f"File {local_path} is empty – skipping deduplication")
        # Write an empty DataFrame back to keep the file present
        df.to_parquet(local_path, index=False)
        return

    # Extract documents
    documents = df['cleaned_text'].tolist()

    # Compute MinHash signatures
    minhashes = compute_minhashes(
        documents,
        num_perm=config.deduplication.num_permutations,
    )

    # Build LSH index and find duplicates
    duplicate_indices = find_duplicates(
        minhashes,
        threshold=config.deduplication.threshold,
        num_perm=config.deduplication.num_permutations,
    )

    # Build deduplicated DataFrame
    keep_mask = [i not in duplicate_indices for i in range(len(documents))]
    deduped_df = df.iloc[keep_mask].reset_index(drop=True)

    # Write back to the same file (overwrite)
    deduped_df.to_parquet(local_path, index=False)
    logger.info(
        f"Removed {len(duplicate_indices)} duplicate documents, "
        f"kept {len(deduped_df)} out of {len(df)}"
    )


def compute_minhashes(
    documents: List[str],
    num_perm: int = 128,
) -> List[MinHash]:
    """
    Compute MinHash signatures for a list of documents.
    Each document is tokenized into shingles (character n‑grams).
    """
    minhashes = []
    for doc in documents:
        shingles = _tokenize_shingles(doc, n=5)
        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        minhashes.append(m)
    logger.debug(f"Computed MinHash signatures for {len(documents)} documents")
    return minhashes


def find_duplicates(
    minhashes: List[MinHash],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> Set[int]:
    """
    Build an LSH index and find all near‑duplicate documents.
    Returns a set of indices to remove (the duplicates).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, mh in enumerate(minhashes):
        lsh.insert(f"doc_{i}", mh)

    duplicates = set()
    for i, mh in enumerate(minhashes):
        candidates = lsh.query(mh)
        for cand in candidates:
            if cand == f"doc_{i}":
                continue
            cand_idx = int(cand.split('_')[1])
            if cand_idx not in duplicates:
                duplicates.add(cand_idx)
    logger.info(f"Found {len(duplicates)} near‑duplicate documents")
    return duplicates


def _tokenize_shingles(text: str, n: int = 5) -> List[str]:
    """
    Split text into character n‑grams (shingles).
    """
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]