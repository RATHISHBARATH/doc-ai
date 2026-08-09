# ============================================================
# DOC AI Vision Service – Crawler Parser
# ============================================================

import json
import logging
import re
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup  # for HTML parsing (optional)

from src.common.config import VisionConfig, get_config
from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient
from src.common.models import Scene  # not directly used, but for type hints

logger = logging.getLogger(__name__)


class CrawlerParser:
    """
    Parser for the automated web crawler.

    Transforms raw fetched data (JSON, HTML, etc.) into structured items,
    validates them, and stores them in MinIO and PostgreSQL.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the parser.

        Args:
            config: VisionConfig instance. If None, loads the global config.
        """
        self.config = config or get_config()
        self.minio = MinIOClient(self.config)
        self.postgres = PostgresClient(self.config)
        self.logger = logging.getLogger(f"{__name__}.CrawlerParser")

    async def parse(self, source: Any, data: Any) -> List[Dict[str, Any]]:
        """
        Parse raw data from a source into a list of structured items.

        Args:
            source: SourceConfig object (or dict) containing 'name', 'type', etc.
            data: Raw data fetched by the fetcher (dict, list, bytes, etc.).

        Returns:
            A list of parsed items, each as a dict with at least 'url', 'metadata'.
            Returns an empty list if parsing fails or no items are found.
        """
        source_name = getattr(source, "name", "unknown")
        source_type = getattr(source, "type", "unknown").lower()

        self.logger.debug(f"Parsing data from {source_name} (type: {source_type})")

        if source_type == "rest":
            return await self._parse_rest(data)
        elif source_type == "s3" or source_type == "http":
            # For raw files (images, videos), we may just store them as-is.
            # But we can also parse if it's a CSV or JSON file.
            # We'll treat bytes as a file to store; no parsing needed.
            self.logger.warning(f"Raw file download not parsed; storing as-is.")
            # We could attempt to interpret based on extension, but for now return empty.
            return []
        elif source_type == "html":
            return await self._parse_html(data)
        else:
            self.logger.error(f"Unsupported source type for parsing: {source_type}")
            return []

    async def _parse_rest(self, data: Union[Dict, List]) -> List[Dict[str, Any]]:
        """
        Parse JSON data from a REST API.

        Expects data to be a list of items or a dict containing a list under a key.
        Each item should have at least an image URL and metadata.

        Supported structures:
        - List of objects: [{"id":123, "poster_path":"/url.jpg", ...}, ...]
        - Dict with results list: {"results": [...]}
        - Paginated responses (handle via config if needed)
        """
        items = []

        # Detect structure
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            # Common keys: 'results', 'data', 'items', 'movies', 'images'
            for key in ["results", "data", "items", "movies", "images", "hits"]:
                if key in data and isinstance(data[key], list):
                    raw_items = data[key]
                    break
            else:
                # If no list found, treat the whole dict as a single item
                self.logger.warning("No list found in REST response; treating as single item")
                raw_items = [data]
        else:
            self.logger.error("REST data is neither list nor dict")
            return []

        # Parse each item
        for raw in raw_items:
            item = self._extract_movie_item(raw)
            if item:
                items.append(item)

        self.logger.info(f"Parsed {len(items)} items from REST response")
        return items

    def _extract_movie_item(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract a movie item from a raw JSON object (e.g., from TMDB).
        Returns None if required fields are missing.
        """
        # TMDB style fields: 'id', 'poster_path', 'backdrop_path', 'title', 'overview'
        # We'll be flexible and try to map common fields.
        try:
            # Try to get image URL
            image_url = None
            if "poster_path" in raw and raw["poster_path"]:
                image_url = f"https://image.tmdb.org/t/p/w500{raw['poster_path']}"
            elif "backdrop_path" in raw and raw["backdrop_path"]:
                image_url = f"https://image.tmdb.org/t/p/w500{raw['backdrop_path']}"
            elif "image_url" in raw:
                image_url = raw["image_url"]
            elif "url" in raw:
                image_url = raw["url"]

            if not image_url:
                self.logger.debug("No image URL found in item; skipping")
                return None

            # Validate URL format
            if not self._is_valid_url(image_url):
                self.logger.warning(f"Invalid image URL: {image_url}")
                return None

            # Extract metadata
            metadata = {
                "source": "crawler",
                "fetched_at": None,  # set later
                "original_data": raw,  # keep full original for reference
            }

            # Copy common fields
            for key in ["id", "title", "overview", "release_date", "vote_average", "vote_count", "popularity"]:
                if key in raw:
                    metadata[key] = raw[key]

            # If we have a movie ID, we can generate a nice name
            if "title" in metadata:
                name = metadata["title"]
            elif "id" in metadata:
                name = f"movie_{metadata['id']}"
            else:
                name = "unknown"

            item = {
                "name": name,
                "url": image_url,
                "metadata": metadata,
            }
            return item

        except Exception as e:
            self.logger.error(f"Error extracting movie item: {e}")
            return None

    async def _parse_html(self, data: Union[str, bytes]) -> List[Dict[str, Any]]:
        """
        Parse HTML data to extract image URLs and metadata.
        Uses BeautifulSoup for parsing.
        """
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                self.logger.error("Failed to decode HTML bytes")
                return []

        soup = BeautifulSoup(data, "html.parser")
        items = []

        # Look for image tags and other relevant content
        # This is a generic parser; for specific sites, we'd need custom logic.
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src")
            if src and self._is_valid_url(src):
                alt = img_tag.get("alt", "")
                title = img_tag.get("title", "")
                item = {
                    "name": alt or title or "image",
                    "url": src,
                    "metadata": {
                        "source": "html_crawler",
                        "alt": alt,
                        "title": title,
                        "fetched_at": None,
                    }
                }
                items.append(item)

        self.logger.info(f"Parsed {len(items)} images from HTML")
        return items

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate a URL.
        """
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Storage methods (called after parsing)
    # ------------------------------------------------------------------

    async def store_items(self, items: List[Dict[str, Any]]) -> int:
        """
        Store parsed items: download images, store metadata, and index.
        Returns the number of items successfully stored.
        """
        stored = 0
        for item in items:
            try:
                # Download the image
                image_data = await self._download_image(item["url"])
                if not image_data:
                    self.logger.warning(f"Failed to download image: {item['url']}")
                    continue

                # Generate a unique ID for the image
                import uuid
                image_id = str(uuid.uuid4())
                remote_path = f"{self.config.storage.raw_images_prefix}{image_id}.jpg"

                # Upload to MinIO
                self.minio.upload_bytes(image_data, remote_path, "image/jpeg")

                # Store metadata in PostgreSQL
                # For simplicity, we'll create a job entry for this image
                job_id = str(uuid.uuid4())
                self.postgres.insert_job(job_id, remote_path, "image")
                self.postgres.start_job(job_id)

                # Optionally, we could run vision tasks on the image here.
                # For now, we just complete the job.
                self.postgres.complete_job(job_id)

                # Store additional metadata as JSONB in the job's metadata field
                # (We could update the job metadata later)

                stored += 1
                self.logger.debug(f"Stored image {image_id} from {item['url']}")

            except Exception as e:
                self.logger.error(f"Error storing item {item.get('name', 'unknown')}: {e}")

        self.logger.info(f"Stored {stored} items out of {len(items)}")
        return stored

    async def _download_image(self, url: str) -> Optional[bytes]:
        """
        Download an image from a URL (async).
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Download failed: {url} -> {resp.status}")
                        return None
                    data = await resp.read()
                    if len(data) == 0:
                        self.logger.warning(f"Downloaded empty file: {url}")
                        return None
                    return data
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return None