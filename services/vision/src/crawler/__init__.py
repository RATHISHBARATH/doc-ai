# ============================================================
# DOC AI Vision Service – Web Crawler Package
# ============================================================

"""
Automated web crawler for continuous data ingestion.

This package provides a scheduled crawler that fetches images, videos,
and metadata from configured public sources (APIs, S3 buckets, etc.).
It includes:
- Scheduler: Cron‑based job scheduling.
- Fetcher: Downloads data from sources with retry logic.
- Parser: Extracts metadata and validates content.
"""

from .scheduler import CrawlerScheduler
from .fetcher import CrawlerFetcher
from .parser import CrawlerParser

__all__ = [
    "CrawlerScheduler",
    "CrawlerFetcher",
    "CrawlerParser",
]