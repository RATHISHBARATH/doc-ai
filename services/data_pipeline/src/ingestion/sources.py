# ============================================================
# DOC AI Data Pipeline – Data Source Helper Module
# ============================================================

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def read_file(file_path: Path, format: str, max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read a file of the given format and return a list of document dictionaries.
    If max_documents is provided, only the first N documents are returned.
    """
    if format == "jsonl":
        return read_jsonl(file_path, max_documents)
    elif format == "parquet":
        return read_parquet(file_path, max_documents)
    elif format == "txt":
        return read_txt(file_path, max_documents)
    else:
        raise ValueError(f"Unsupported format: {format}")


def read_jsonl(file_path: Path, max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read a JSONL file (each line is a JSON object).
    Returns a list of dictionaries, where each dict contains the document data.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_documents and i >= max_documents:
                break
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON line {i+1}: {e}")
                continue
    logger.info(f"Read {len(documents)} documents from JSONL file")
    return documents


def read_parquet(file_path: Path, max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read a Parquet file and return a list of row dictionaries.
    If max_documents is provided, only the first N rows are returned.
    """
    # Use pandas to read the Parquet file
    df = pd.read_parquet(file_path)
    if max_documents:
        df = df.head(max_documents)
    # Convert to list of dicts
    documents = df.to_dict(orient='records')
    logger.info(f"Read {len(documents)} documents from Parquet file")
    return documents


def read_txt(file_path: Path, max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read a plain text file and return a list of documents.
    Each document is a dict with a 'text' key containing the entire file content.
    If max_documents is provided, it is ignored for plain text (only one document).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    documents = [{"text": text}]
    logger.info(f"Read 1 document from plain text file")
    return documents


def truncate_file(file_path: Path, format: str, max_documents: int) -> Path:
    """
    Truncate a file to keep only the first `max_documents` documents.
    Returns the path to the truncated file (overwrites the original).
    """
    if format == "jsonl":
        return truncate_jsonl(file_path, max_documents)
    elif format == "parquet":
        return truncate_parquet(file_path, max_documents)
    elif format == "txt":
        # Plain text cannot be truncated by document count; we keep the whole file.
        logger.warning("Truncation not supported for plain text format.")
        return file_path
    else:
        raise ValueError(f"Unsupported format for truncation: {format}")


def truncate_jsonl(file_path: Path, max_documents: int) -> Path:
    """
    Truncate a JSONL file to the first `max_documents` lines.
    Overwrites the original file.
    """
    temp_path = file_path.with_suffix(".tmp")
    with open(file_path, 'r', encoding='utf-8') as src:
        with open(temp_path, 'w', encoding='utf-8') as dst:
            for i, line in enumerate(src):
                if i >= max_documents:
                    break
                dst.write(line)
    # Replace the original file with the truncated one
    os.replace(temp_path, file_path)
    logger.info(f"Truncated JSONL file to {max_documents} documents")
    return file_path


def truncate_parquet(file_path: Path, max_documents: int) -> Path:
    """
    Truncate a Parquet file to the first `max_documents` rows.
    Overwrites the original file.
    """
    df = pd.read_parquet(file_path)
    df = df.head(max_documents)
    df.to_parquet(file_path, engine='pyarrow', compression='snappy')
    logger.info(f"Truncated Parquet file to {max_documents} rows")
    return file_path