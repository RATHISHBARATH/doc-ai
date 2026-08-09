# ============================================================
# DOC AI Data Pipeline – Text Cleaning Module (Corrected)
# ============================================================

import logging
import re
import os
from typing import List, Dict, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ftfy
from langdetect import detect, LangDetectException

from src.common.minio_client import MinIOClient
from src.common.config import Config

logger = logging.getLogger(__name__)


def clean_data(config: Config, client: MinIOClient) -> None:
    """
    Main entry point: clean all raw data from MinIO and write cleaned data back.
    """
    raw_objects = client.list_objects(prefix="raw/", recursive=True)
    logger.info(f"Found {len(raw_objects)} raw objects to clean")

    for obj in raw_objects:
        local_path = f"/tmp/raw_{obj.replace('/', '_')}"
        client.download_file(obj, local_path)

        # Clean the file and get the cleaned DataFrame
        cleaned_df = clean_file(local_path, config)

        # Overwrite the local file with cleaned Parquet data
        cleaned_df.to_parquet(local_path, engine='pyarrow', compression='snappy')

        # Upload cleaned data to MinIO (replace raw/ with cleaned/)
        cleaned_remote = obj.replace("raw/", "cleaned/").replace(".jsonl", ".parquet").replace(".json", ".parquet")
        if not cleaned_remote.endswith(".parquet"):
            cleaned_remote += ".parquet"
        client.upload_file(local_path, cleaned_remote)

        # Clean up local files
        os.remove(local_path)


def clean_file(local_path: str, config: Config) -> pd.DataFrame:
    """
    Read a file (JSONL, Parquet, etc.) and clean each document.
    Returns a DataFrame with the cleaned text.
    """
    # Determine file format from extension
    if local_path.endswith(".jsonl") or local_path.endswith(".json"):
        df = pd.read_json(local_path, lines=True)
    elif local_path.endswith(".parquet"):
        df = pd.read_parquet(local_path)
    else:
        # Fallback: read as plain text
        with open(local_path, 'r') as f:
            text = f.read()
        df = pd.DataFrame({"text": [text]})

    # Ensure we have a 'text' column; if not, try to find it
    if 'text' not in df.columns:
        # If there's a 'content' or 'body' column, rename it
        for col in ['content', 'body', 'document']:
            if col in df.columns:
                df['text'] = df[col]
                break
        else:
            # If no text column, treat the entire DataFrame as text?
            # We'll raise an error for now.
            raise ValueError("Input data must contain a 'text' column (or 'content', 'body', 'document')")

    # Apply cleaning to each row
    df['cleaned_text'] = df['text'].apply(lambda x: clean_document(x, config))
    # Filter out rows where cleaning returned None or empty string
    df = df[df['cleaned_text'].notna() & (df['cleaned_text'].str.strip() != "")]
    df = df.drop(columns=['text'])  # remove original text column

    logger.info(f"Cleaned {len(df)} documents from {local_path}")
    return df


def clean_document(text: str, config: Config) -> Optional[str]:
    """
    Apply all cleaning steps to a single document.
    Returns None if the document should be filtered out.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Fix Unicode
    if config.cleaning.fix_unicode:
        text = ftfy.fix_text(text)

    # 2. Normalize whitespace
    if config.cleaning.normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    # 3. Remove control characters
    if config.cleaning.remove_control_chars:
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    # 4. Filter language (optional)
    if config.cleaning.filter_language:
        try:
            lang = detect(text)
            if lang != config.cleaning.filter_language:
                return None
        except LangDetectException:
            # If language detection fails, keep the document
            pass

    # 5. Scrub PII
    if config.cleaning.scrub_pii:
        text = scrub_pii(text)

    # If after cleaning the text is empty, return None
    if not text or text.strip() == "":
        return None

    return text


def scrub_pii(text: str) -> str:
    """
    Remove or mask common PII patterns (emails, phone numbers, IP addresses).
    """
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b(\+?[0-9]{1,3}[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[REDACTED_IP]', text)
    return text