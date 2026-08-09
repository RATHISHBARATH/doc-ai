# ============================================================
# DOC AI Vision Service – Dataset Ingestion Package
# ============================================================

"""
Large‑scale dataset ingestion for multimodal training.

This package provides tools to download, validate, and preprocess
large datasets (e.g., 1M+ movie images with metadata) for training.
It includes:
- Downloader: Fetches images and metadata from configured sources.
- Preprocessor: Resizes, normalizes, and formats images for training.
"""

from .downloader import DatasetDownloader
from .preprocessor import DatasetPreprocessor

__all__ = [
    "DatasetDownloader",
    "DatasetPreprocessor",
]