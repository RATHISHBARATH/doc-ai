# ============================================================
# DOC AI Vision Service – Deduplicator
# ============================================================

import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash  # pip install ImageHash

from src.common.config import VisionConfig, get_config
from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Deduplication engine for images and metadata.

    Supports:
    - Exact deduplication via SHA‑256 hash of image bytes.
    - Perceptual deduplication via pHash (perceptual hash) to find near‑duplicates.
    - Metadata‑based deduplication (e.g., same source URL, same title, etc.).
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the deduplicator.

        Args:
            config: VisionConfig instance. If None, loads the global config.
        """
        self.config = config or get_config()
        self.minio = MinIOClient(self.config)
        self.postgres = PostgresClient(self.config)
        self.logger = logging.getLogger(f"{__name__}.Deduplicator")

        # Cache for hashes (to avoid re‑computing)
        self._image_hashes: Dict[str, str] = {}  # path -> hash
        self._perceptual_hashes: Dict[str, str] = {}  # path -> pHash

    # ------------------------------------------------------------------
    # Exact deduplication (SHA‑256)
    # ------------------------------------------------------------------

    def compute_sha256(self, data: bytes) -> str:
        """Compute SHA‑256 hash of image bytes."""
        return hashlib.sha256(data).hexdigest()

    def compute_sha256_from_minio(self, remote_path: str) -> str:
        """Compute SHA‑256 hash of an image stored in MinIO."""
        data = self.minio.download_bytes(remote_path)
        return self.compute_sha256(data)

    def is_exact_duplicate(self, image_data: bytes, known_hashes: Set[str]) -> bool:
        """Check if an image is an exact duplicate given a set of known hashes."""
        h = self.compute_sha256(image_data)
        return h in known_hashes

    # ------------------------------------------------------------------
    # Perceptual deduplication (pHash)
    # ------------------------------------------------------------------

    def compute_phash(self, image_data: bytes, hash_size: int = 8) -> str:
        """
        Compute perceptual hash (pHash) of an image.

        Args:
            image_data: Raw image bytes.
            hash_size: Size of the hash (8 = 64 bits, 16 = 256 bits, etc.)

        Returns:
            Hexadecimal string representation of the hash.
        """
        try:
            img = Image.open(BytesIO(image_data))
            # Convert to RGB to handle grayscale/alpha
            img = img.convert("RGB")
            # Compute pHash
            phash = imagehash.phash(img, hash_size=hash_size)
            return str(phash)
        except Exception as e:
            self.logger.warning(f"Failed to compute pHash: {e}")
            return ""

    def compute_phash_from_minio(self, remote_path: str, hash_size: int = 8) -> str:
        """Compute pHash of an image stored in MinIO."""
        data = self.minio.download_bytes(remote_path)
        return self.compute_phash(data, hash_size)

    def is_perceptual_duplicate(self, image_data: bytes, known_phashes: Set[str], threshold: int = 10) -> bool:
        """
        Check if an image is a perceptual duplicate given a set of known pHashes.

        Args:
            image_data: Raw image bytes.
            known_phashes: Set of known pHash strings.
            threshold: Maximum Hamming distance to consider as duplicate.

        Returns:
            True if the image is a perceptual duplicate.
        """
        if not known_phashes:
            return False

        phash = self.compute_phash(image_data)
        if not phash:
            return False

        # Compare pHash with each known hash
        for known in known_phashes:
            try:
                # Convert to imagehash objects
                h1 = imagehash.hex_to_hash(phash)
                h2 = imagehash.hex_to_hash(known)
                distance = h1 - h2  # Hamming distance
                if distance <= threshold:
                    return True
            except Exception as e:
                self.logger.debug(f"Error comparing pHashes: {e}")
                continue
        return False

    # ------------------------------------------------------------------
    # Metadata‑based deduplication
    # ------------------------------------------------------------------

    def is_metadata_duplicate(self, metadata: Dict[str, Any], known_metadata_keys: List[str]) -> bool:
        """
        Check if metadata matches any known record based on key values.

        Args:
            metadata: Metadata dict of the new item.
            known_metadata_keys: List of keys to compare (e.g., ['title', 'release_date']).

        Returns:
            True if a duplicate is found.
        """
        if not known_metadata_keys:
            return False

        # Query PostgreSQL for existing records with matching key values
        # This is a simplified example; real implementation would use a proper query.
        # We'll just check if we have any record with the same values for the given keys.
        conditions = []
        params = []
        for key in known_metadata_keys:
            if key in metadata:
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(metadata[key])

        if not conditions:
            return False

        query = f"""
            SELECT COUNT(*) FROM vision_processing_jobs
            WHERE {' AND '.join(conditions)}
        """
        result = self.postgres.execute_query(query, tuple(params), fetch=True)
        if result and result[0][0] > 0:
            self.logger.debug(f"Metadata duplicate found for keys: {known_metadata_keys}")
            return True
        return False

    # ------------------------------------------------------------------
    # Full deduplication pipeline
    # ------------------------------------------------------------------

    async def deduplicate_items(
        self,
        items: List[Dict[str, Any]],
        use_exact: bool = True,
        use_perceptual: bool = True,
        use_metadata: bool = False,
        metadata_keys: Optional[List[str]] = None,
        known_hashes: Optional[Set[str]] = None,
        known_phashes: Optional[Set[str]] = None,
        threshold: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate a list of items (image URLs with metadata).

        Args:
            items: List of item dicts, each with 'url' and optionally 'metadata'.
            use_exact: Whether to perform exact SHA‑256 deduplication.
            use_perceptual: Whether to perform perceptual pHash deduplication.
            use_metadata: Whether to perform metadata‑based deduplication.
            metadata_keys: Keys to use for metadata deduplication.
            known_hashes: Pre‑computed set of known SHA‑256 hashes.
            known_phashes: Pre‑computed set of known pHashes.
            threshold: Hamming distance threshold for perceptual deduplication.

        Returns:
            List of items that are not duplicates.
        """
        self.logger.info(f"Deduplicating {len(items)} items (exact={use_exact}, perceptual={use_perceptual}, metadata={use_metadata})")

        # If we don't have pre‑computed hashes, we'll build them from existing items
        if use_exact and known_hashes is None:
            known_hashes = self._load_existing_hashes(exact=True)
        if use_perceptual and known_phashes is None:
            known_phashes = self._load_existing_hashes(exact=False)

        unique_items = []
        duplicates_found = 0

        for item in items:
            is_duplicate = False

            # We need to download the image to compute hashes
            # To avoid downloading twice, we download once and cache.
            image_data = None
            hash_sha256 = None
            hash_phash = None

            if use_exact or use_perceptual:
                # Download image
                import aiohttp
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(item["url"], timeout=30) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                            else:
                                self.logger.warning(f"Failed to download {item['url']} for deduplication")
                                # Skip this item? We'll treat as unique to avoid false negatives.
                                is_duplicate = False
                except Exception as e:
                    self.logger.warning(f"Error downloading {item['url']} for deduplication: {e}")
                    is_duplicate = False

            # Exact deduplication
            if use_exact and image_data and not is_duplicate:
                hash_sha256 = self.compute_sha256(image_data)
                if hash_sha256 in known_hashes:
                    is_duplicate = True
                    duplicates_found += 1
                    self.logger.debug(f"Exact duplicate found: {item['url']} (hash: {hash_sha256[:8]}...)")

            # Perceptual deduplication
            if use_perceptual and image_data and not is_duplicate:
                hash_phash = self.compute_phash(image_data)
                if hash_phash and self.is_perceptual_duplicate(image_data, known_phashes, threshold):
                    is_duplicate = True
                    duplicates_found += 1
                    self.logger.debug(f"Perceptual duplicate found: {item['url']} (phash: {hash_phash[:8]}...)")

            # Metadata deduplication
            if use_metadata and not is_duplicate and metadata_keys:
                if self.is_metadata_duplicate(item.get("metadata", {}), metadata_keys):
                    is_duplicate = True
                    duplicates_found += 1
                    self.logger.debug(f"Metadata duplicate found: {item['url']}")

            if not is_duplicate:
                unique_items.append(item)
                # Add hashes to known sets for future comparisons
                if use_exact and hash_sha256:
                    known_hashes.add(hash_sha256)
                if use_perceptual and hash_phash:
                    known_phashes.add(hash_phash)

        self.logger.info(f"Deduplication complete: {len(unique_items)} unique, {duplicates_found} duplicates removed")
        return unique_items

    # ------------------------------------------------------------------
    # Helper: Load existing hashes from PostgreSQL / MinIO
    # ------------------------------------------------------------------

    def _load_existing_hashes(self, exact: bool = True) -> Set[str]:
        """
        Load existing hashes from the database (or compute from stored images).

        Returns:
            A set of hash strings.
        """
        # For a real implementation, we would store hashes in PostgreSQL when images are ingested.
        # For simplicity, we'll return an empty set and rely on the caller to provide them.
        self.logger.debug("Loading existing hashes – not implemented yet; returning empty set.")
        return set()

    # ------------------------------------------------------------------
    # Utility: Compute hash from image URL without storing
    # ------------------------------------------------------------------

    async def compute_hashes_from_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Compute both SHA‑256 and pHash from an image URL.
        Returns (sha256, phash).
        """
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        return None, None
                    data = await resp.read()
                    sha = self.compute_sha256(data)
                    phash = self.compute_phash(data)
                    return sha, phash
        except Exception as e:
            self.logger.error(f"Error computing hashes from {url}: {e}")
            return None, None