# ============================================================
# DOC AI Data Pipeline – Quality Scoring Module
# ============================================================

import logging
import re
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from src.common.minio_client import MinIOClient
from src.common.config import Config

logger = logging.getLogger(__name__)

# Common English stop words (small set for speed)
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'without', 'about', 'against',
    'between', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'but', 'now'
}


def quality_score(config: Config, client: MinIOClient) -> None:
    """
    Main entry point: apply quality filtering to deduplicated data.
    """
    deduped_objects = client.list_objects(prefix="deduped/", recursive=True)
    logger.info(f"Found {len(deduped_objects)} deduped objects to filter")

    for obj in deduped_objects:
        # Download the deduplicated file
        local_path = f"/tmp/deduped_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)

        # Apply quality filtering
        filtered_df = filter_file(local_path, config)

        # Upload filtered data to MinIO
        filtered_remote = obj.replace("deduped/", "filtered/")
        client.upload_file(local_path, filtered_remote)

        # Clean up
        import os
        os.remove(local_path)


def filter_file(local_path: str, config: Config) -> pd.DataFrame:
    """
    Read a Parquet file, score each document, and keep only those above the threshold.
    """
    df = pd.read_parquet(local_path)

    if 'cleaned_text' not in df.columns:
        raise ValueError("Input Parquet must contain 'cleaned_text' column")

    # Compute quality score for each document
    df['quality_score'] = df['cleaned_text'].apply(
        lambda x: compute_quality_score(x, config)
    )

    # Apply threshold (we can keep documents with score > 0.6, configurable)
    threshold = 0.6  # This could be made configurable
    filtered_df = df[df['quality_score'] >= threshold].reset_index(drop=True)

    logger.info(f"Kept {len(filtered_df)}/{len(df)} documents after quality filtering")
    return filtered_df


def compute_quality_score(text: str, config: Config) -> float:
    """
    Compute a quality score (0–1) for a single document.
    Higher score = higher quality.
    """
    if not text or not isinstance(text, str):
        return 0.0

    # 1. Length score
    words = text.split()
    word_count = len(words)
    length_score = length_scoring(word_count, config.quality_filter.min_words, config.quality_filter.max_words)

    # 2. Punctuation ratio score
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    if word_count > 0:
        punctuation_ratio = punctuation_count / word_count
    else:
        punctuation_ratio = 0.0
    punctuation_score = 1.0 - min(1.0, punctuation_ratio / config.quality_filter.max_punctuation_ratio)

    # 3. Stop word ratio score
    stop_count = sum(1 for word in words if word.lower() in STOP_WORDS)
    if word_count > 0:
        stop_ratio = stop_count / word_count
    else:
        stop_ratio = 0.0
    # We want a minimum stop word ratio (to avoid nonsensical text)
    stop_score = 1.0 if stop_ratio >= config.quality_filter.min_stop_word_ratio else 0.0

    # Combine scores (weighted average)
    # Weights: length 0.4, punctuation 0.3, stop words 0.3
    total_score = 0.4 * length_score + 0.3 * punctuation_score + 0.3 * stop_score

    # Optional: add perplexity scoring if a model is available
    # if config.quality_filter.scorer_model_path:
    #     perplexity_score = compute_perplexity(text, config.quality_filter.scorer_model_path)
    #     total_score = 0.5 * total_score + 0.5 * perplexity_score

    return min(1.0, max(0.0, total_score))


def length_scoring(word_count: int, min_words: int, max_words: int) -> float:
    """
    Score based on document length.
    Returns 1.0 if within [min_words, max_words], 0.0 if outside.
    """
    if min_words <= word_count <= max_words:
        return 1.0
    else:
        # Allow partial credit near the boundaries
        if word_count < min_words:
            return max(0.0, word_count / min_words)
        else:  # word_count > max_words
            # Linearly decay from 1.0 to 0.0 as length approaches 2*max_words
            excess = word_count - max_words
            if excess >= max_words:
                return 0.0
            return 1.0 - (excess / max_words)