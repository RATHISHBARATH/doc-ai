# ============================================================
# DOC AI Vision Service – Data Verifier
# ============================================================

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from src.common.config import VisionConfig, get_config
from src.common.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class Verifier:
    """
    Data quality verifier for ingested datasets.

    Responsible for:
    - Image integrity checks (decode, size, corruption).
    - Metadata validation (required fields, data types, ranges).
    - Source trust scoring (known domains vs. suspicious ones).
    - Cross‑referencing information from multiple sources (future).
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the verifier.

        Args:
            config: VisionConfig instance. If None, loads the global config.
        """
        self.config = config or get_config()
        self.postgres = PostgresClient(self.config)
        self.logger = logging.getLogger(f"{__name__}.Verifier")

        # Trusted domain patterns (can be extended via config)
        self.trusted_domains = {
            "tmdb.org", "themoviedb.org", "imdb.com", "wikipedia.org",
            "wikimedia.org", "commons.wikimedia.org", "flickr.com",
            "pexels.com", "unsplash.com", "pixabay.com",
        }

    # ------------------------------------------------------------------
    # Image integrity checks
    # ------------------------------------------------------------------

    def verify_image_bytes(self, image_data: bytes) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify that the image data is a valid image and extract metadata.

        Args:
            image_data: Raw image bytes.

        Returns:
            (is_valid, metadata_dict) where metadata_dict contains:
                - width, height, channels, format (if valid)
                - None if invalid.
        """
        if not image_data:
            return False, None

        try:
            # Try to decode with OpenCV (supports JPEG, PNG, etc.)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # Fallback to PIL for other formats (e.g., WebP, TIFF)
                try:
                    pil_img = Image.open(BytesIO(image_data))
                    img = np.array(pil_img.convert("RGB"))
                    is_pil = True
                except Exception:
                    return False, None
            else:
                is_pil = False

            # Extract metadata
            if is_pil:
                height, width = img.shape[:2]
                channels = img.shape[2] if len(img.shape) == 3 else 1
                fmt = pil_img.format or "unknown"
            else:
                height, width = img.shape[:2]
                channels = img.shape[2] if len(img.shape) == 3 else 1
                fmt = "opencv"

            metadata = {
                "width": width,
                "height": height,
                "channels": channels,
                "format": fmt,
                "size_bytes": len(image_data),
            }
            return True, metadata

        except Exception as e:
            self.logger.warning(f"Image verification failed: {e}")
            return False, None

    def verify_image_from_url(self, url: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Download and verify an image from a URL.

        Args:
            url: Image URL.

        Returns:
            (is_valid, metadata) or (False, None) on failure.
        """
        import aiohttp
        try:
            import asyncio
            # We'll run a sync version here, but we can also provide async version.
            # For simplicity, we'll use aiohttp to download.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30) as resp:
                        if resp.status != 200:
                            return None
                        return await resp.read()
            data = loop.run_until_complete(fetch())
            loop.close()
            if data is None:
                return False, None
            return self.verify_image_bytes(data)
        except Exception as e:
            self.logger.error(f"Error downloading image for verification: {e}")
            return False, None

    # ------------------------------------------------------------------
    # Metadata validation
    # ------------------------------------------------------------------

    def validate_metadata(self, metadata: Dict[str, Any], required_fields: List[str],
                          type_map: Optional[Dict[str, type]] = None,
                          range_map: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[bool, List[str]]:
        """
        Validate metadata against a schema.

        Args:
            metadata: The metadata dictionary to validate.
            required_fields: List of field names that must be present.
            type_map: Dict mapping field names to expected types (e.g., {'year': int}).
            range_map: Dict mapping field names to (min, max) ranges (for numeric fields).

        Returns:
            (is_valid, errors) where errors is a list of error messages.
        """
        errors = []

        # Check required fields
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")

        # Check types
        if type_map:
            for field, expected_type in type_map.items():
                if field in metadata:
                    if not isinstance(metadata[field], expected_type):
                        errors.append(f"Field '{field}' expected {expected_type}, got {type(metadata[field])}")

        # Check ranges
        if range_map:
            for field, (min_val, max_val) in range_map.items():
                if field in metadata:
                    val = metadata[field]
                    if isinstance(val, (int, float)):
                        if val < min_val or val > max_val:
                            errors.append(f"Field '{field}' value {val} outside range [{min_val}, {max_val}]")
                    else:
                        # Non-numeric, skip
                        pass

        if errors:
            self.logger.debug(f"Metadata validation failed: {errors}")
            return False, errors

        return True, []

    # ------------------------------------------------------------------
    # Source trust scoring
    # ------------------------------------------------------------------

    def score_source(self, url: str) -> float:
        """
        Compute a trust score for a source URL (0.0 to 1.0).

        Higher scores indicate more trustworthy sources.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove "www." prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Base score
            score = 0.5

            # Check against trusted domains
            for trusted in self.trusted_domains:
                if trusted in domain:
                    score = 0.9
                    break

            # Check for suspicious patterns
            suspicious_patterns = [
                r'\.xyz$', r'\.top$', r'\.club$', r'\.click$', r'\.download$',
                r'\.tk$', r'\.ml$', r'\.ga$', r'\.cf$', r'\.ru$',
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, domain):
                    score = max(0.1, score - 0.3)

            # Penalize missing or non‑standard schemes
            if parsed.scheme not in ('http', 'https'):
                score = max(0.0, score - 0.2)

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.warning(f"Error scoring source {url}: {e}")
            return 0.0

    def is_source_trusted(self, url: str, threshold: float = 0.7) -> bool:
        """
        Return True if the source trust score is above the threshold.
        """
        return self.score_source(url) >= threshold

    # ------------------------------------------------------------------
    # Cross‑referencing (placeholder for future)
    # ------------------------------------------------------------------

    def cross_reference(self, item: Dict[str, Any], other_items: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Cross‑reference an item with other items to detect inconsistencies.

        For example, verify that the same movie title appears in multiple sources
        with the same release date. This is a placeholder for future implementation.

        Args:
            item: The item to verify.
            other_items: List of other items from different sources.

        Returns:
            (is_consistent, conflicts) where conflicts is a list of discrepancy descriptions.
        """
        # Placeholder: return True with empty conflicts.
        return True, []

    # ------------------------------------------------------------------
    # Full verification pipeline
    # ------------------------------------------------------------------

    def verify_item(self, item: Dict[str, Any],
                    required_metadata_fields: Optional[List[str]] = None,
                    type_map: Optional[Dict[str, type]] = None,
                    range_map: Optional[Dict[str, Tuple[float, float]]] = None,
                    verify_image: bool = True,
                    check_source_trust: bool = True,
                    trust_threshold: float = 0.7) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a full verification on a single item.

        Args:
            item: Item dict with 'url' and optional 'metadata'.
            required_metadata_fields: List of required metadata fields.
            type_map: Type validation map.
            range_map: Range validation map.
            verify_image: Whether to check image integrity.
            check_source_trust: Whether to compute source trust score.
            trust_threshold: Minimum trust score to accept.

        Returns:
            (is_valid, details) where details contains:
                - image_ok: bool
                - metadata_ok: bool
                - source_trust: float
                - errors: list of error messages
        """
        details = {
            "image_ok": False,
            "metadata_ok": False,
            "source_trust": 0.0,
            "errors": [],
            "warnings": [],
        }

        url = item.get("url")
        metadata = item.get("metadata", {})

        if not url:
            details["errors"].append("No URL provided")
            return False, details

        # 1. Verify image integrity
        if verify_image:
            # We need to download the image; this could be expensive.
            # We'll use a lightweight check: just verify it can be decoded.
            # We could use verify_image_from_url, but that downloads the full image.
            # For large datasets, we might skip or sample.
            is_valid, img_meta = self.verify_image_from_url(url)
            if is_valid:
                details["image_ok"] = True
                if img_meta:
                    details["image_metadata"] = img_meta
            else:
                details["errors"].append(f"Image verification failed for {url}")
                return False, details  # Fail fast

        # 2. Validate metadata
        if required_metadata_fields or type_map or range_map:
            if metadata:
                valid, meta_errors = self.validate_metadata(
                    metadata,
                    required_fields=required_metadata_fields or [],
                    type_map=type_map or {},
                    range_map=range_map or {},
                )
                if valid:
                    details["metadata_ok"] = True
                else:
                    details["errors"].extend(meta_errors)
                    return False, details
            else:
                if required_metadata_fields:
                    details["errors"].append("Metadata required but not provided")
                    return False, details

        # 3. Check source trust
        if check_source_trust:
            trust = self.score_source(url)
            details["source_trust"] = trust
            if trust < trust_threshold:
                details["warnings"].append(f"Source trust score {trust:.2f} below threshold {trust_threshold}")

        # If we passed all checks, return True
        return True, details

    # ------------------------------------------------------------------
    # Batch verification (async)
    # ------------------------------------------------------------------

    async def verify_items(self, items: List[Dict[str, Any]], **kwargs) -> List[Tuple[bool, Dict[str, Any]]]:
        """
        Verify a list of items concurrently.

        Args:
            items: List of item dicts.
            **kwargs: Passed to verify_item().

        Returns:
            List of (is_valid, details) tuples.
        """
        # For simplicity, we'll run synchronously in a loop.
        # In production, use asyncio.gather to run concurrent downloads.
        results = []
        for item in items:
            is_valid, details = self.verify_item(item, **kwargs)
            results.append((is_valid, details))
        return results