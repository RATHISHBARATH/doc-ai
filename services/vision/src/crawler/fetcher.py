# ============================================================
# DOC AI Vision Service – Crawler Fetcher
# ============================================================

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp
from aiohttp import ClientTimeout, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.config import VisionConfig, get_config
from src.common.minio_client import MinIOClient
from src.crawler.parser import CrawlerParser

logger = logging.getLogger(__name__)


class CrawlerFetcher:
    """
    Fetcher for the automated web crawler.

    Handles fetching data from different source types:
    - REST APIs (JSON, XML, etc.)
    - S3 buckets (using MinIO client)
    - Generic HTTP downloads (images, videos, etc.)
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the fetcher.

        Args:
            config: VisionConfig instance. If None, loads the global config.
        """
        self.config = config or get_config()
        self.minio = MinIOClient(self.config)
        self.logger = logging.getLogger(f"{__name__}.CrawlerFetcher")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self.logger.info("Fetcher aiohttp session closed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def fetch(self, source: Any) -> Optional[Union[Dict, List, bytes]]:
        """
        Fetch data from a source.

        Args:
            source: A SourceConfig object (from config) or a dict with 'url', 'type'.

        Returns:
            Parsed data (dict/list for REST, bytes for raw downloads) or None on failure.
        """
        source_type = source.type.lower()
        url = source.url
        self.logger.debug(f"Fetching from {source.name} ({url})")

        if source_type == "rest":
            return await self._fetch_rest(url)
        elif source_type == "s3":
            return await self._fetch_s3(url)
        elif source_type == "http" or source_type == "download":
            return await self._fetch_http(url)
        else:
            self.logger.error(f"Unsupported source type: {source_type}")
            return None

    async def _fetch_rest(self, url: str) -> Optional[Union[Dict, List]]:
        """
        Fetch data from a REST API endpoint (expects JSON response).
        """
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    self.logger.warning(f"REST fetch returned status {resp.status} for {url}")
                    return None
                data = await resp.json()
                self.logger.debug(f"Fetched JSON data from {url} (size: {len(str(data))} chars)")
                return data
        except Exception as e:
            self.logger.error(f"REST fetch error for {url}: {e}")
            raise

    async def _fetch_s3(self, url: str) -> Optional[bytes]:
        """
        Fetch data from an S3 bucket (via MinIO).
        The URL should be in the format: "minio://bucket/path/to/object"
        """
        if not url.startswith("minio://"):
            self.logger.error(f"Invalid S3 URL: {url} (must start with 'minio://')")
            return None

        # Remove the scheme
        object_path = url[8:]  # remove "minio://"
        try:
            data = self.minio.download_bytes(object_path)
            self.logger.debug(f"Downloaded {len(data)} bytes from MinIO: {object_path}")
            return data
        except Exception as e:
            self.logger.error(f"MinIO download error for {object_path}: {e}")
            raise

    async def _fetch_http(self, url: str) -> Optional[bytes]:
        """
        Download raw data (e.g., image, video) from an HTTP URL.
        """
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    self.logger.warning(f"HTTP fetch returned status {resp.status} for {url}")
                    return None
                data = await resp.read()
                self.logger.debug(f"Downloaded {len(data)} bytes from {url}")
                return data
        except Exception as e:
            self.logger.error(f"HTTP fetch error for {url}: {e}")
            raise

    async def fetch_batch(self, sources: List[Any]) -> List[Optional[Any]]:
        """
        Fetch from multiple sources concurrently.

        Args:
            sources: List of source configs.

        Returns:
            List of fetched data (or None for failures) in the same order.
        """
        tasks = [self.fetch(src) for src in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exceptions and log them
        clean_results = []
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Batch fetch failed: {res}")
                clean_results.append(None)
            else:
                clean_results.append(res)
        return clean_results