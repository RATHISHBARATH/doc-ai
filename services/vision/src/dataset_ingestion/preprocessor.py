# ============================================================
# DOC AI Vision Service – Dataset Preprocessor
# ============================================================

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from io import BytesIO

from src.common.config import VisionConfig, get_config
from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """
    Preprocessor for large‑scale multimodal datasets.

    Handles image preprocessing tasks such as:
    - Resizing to a target size (e.g., 224x224).
    - Normalization (mean/std, or 0‑1 scaling).
    - Conversion to RGB and safe formats.
    - Optional augmentations (flips, rotations, color jitter).
    - Storing processed images in MinIO and updating metadata.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
        output_format: str = "jpg",
        quality: int = 90,
    ):
        """
        Initialize the preprocessor.

        Args:
            target_size: Output size as (width, height).
            normalize: Whether to apply normalization.
            mean: Mean values for normalization (RGB order).
            std: Standard deviation values for normalization.
            augment: Whether to apply augmentations (useful for training).
            output_format: Output format ('jpg', 'png', 'npz').
            quality: JPEG quality (if output_format is 'jpg').
        """
        self.config = get_config()
        self.minio = MinIOClient(self.config)
        self.postgres = PostgresClient(self.config)
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.augment = augment
        self.output_format = output_format.lower()
        self.quality = quality
        self.logger = logging.getLogger(f"{__name__}.DatasetPreprocessor")

    # ------------------------------------------------------------------
    # Core preprocessing pipeline
    # ------------------------------------------------------------------

    def _process_one(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Preprocess a single image from raw bytes.

        Returns:
            Preprocessed image bytes in the specified output format,
            or None if processing fails.
        """
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                self.logger.warning("Failed to decode image")
                return None

            # Convert to RGB (if needed)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize
            img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

            # Apply augmentations (optional)
            if self.augment:
                img_resized = self._augment(img_resized)

            # Normalize (optional)
            if self.normalize:
                img_resized = self._normalize(img_resized)

            # Encode to output format
            if self.output_format == "jpg":
                success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not success:
                    self.logger.warning("Failed to encode JPEG")
                    return None
                return encoded.tobytes()
            elif self.output_format == "png":
                success, encoded = cv2.imencode(".png", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                if not success:
                    return None
                return encoded.tobytes()
            elif self.output_format == "npz":
                # Store as compressed numpy array
                buffer = BytesIO()
                np.savez_compressed(buffer, image=img_resized.astype(np.float32) if self.normalize else img_resized)
                return buffer.getvalue()
            else:
                self.logger.error(f"Unsupported output format: {self.output_format}")
                return None
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            return None

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean/std.
        Assumes input is RGB with values in [0, 255].
        Returns float32 array in [0, 1] range with normalization applied.
        """
        img_float = img.astype(np.float32) / 255.0
        for c in range(3):
            img_float[:, :, c] = (img_float[:, :, c] - self.mean[c]) / self.std[c]
        return img_float

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        Apply simple augmentations: random horizontal flip and rotation.
        """
        import random
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        # Random rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        return img

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def process_item(self, item: Dict[str, Any]) -> bool:
        """
        Process a single item: download raw image, preprocess, store in MinIO.

        Args:
            item: Dict containing 'url' and optionally 'metadata'.

        Returns:
            True if successful, False otherwise.
        """
        try:
            url = item["url"]
            metadata = item.get("metadata", {})

            # Download raw image (use aiohttp)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Download failed for {url}: status {resp.status}")
                        return False
                    raw_data = await resp.read()

            # Preprocess
            processed_bytes = self._process_one(raw_data)
            if processed_bytes is None:
                return False

            # Store processed image in MinIO
            image_id = str(uuid.uuid4())
            remote_path = f"{self.config.storage.processed_images_prefix}{image_id}.{self.output_format}"
            self.minio.upload_bytes(processed_bytes, remote_path, f"image/{self.output_format}")

            # Update metadata in PostgreSQL (link to original job)
            # We'll find the job that corresponds to this original URL.
            # For simplicity, we create a new job entry for the processed image.
            job_id = str(uuid.uuid4())
            self.postgres.insert_job(job_id, remote_path, "image")
            self.postgres.start_job(job_id)
            self.postgres.execute_query(
                "UPDATE vision_processing_jobs SET metadata = metadata || %s WHERE job_id = %s",
                (json.dumps({"original_url": url, "processed": True, "metadata": metadata}),),
                commit=True,
            )
            self.postgres.complete_job(job_id)

            self.logger.debug(f"Processed and stored image {image_id} from {url}")
            return True
        except Exception as e:
            self.logger.error(f"Error processing item {item.get('url', 'unknown')}: {e}")
            return False

    async def process_dataset(self, items: List[Dict[str, Any]], max_concurrent: int = 8) -> int:
        """
        Process a list of items concurrently.

        Args:
            items: List of item dicts with 'url' and optional 'metadata'.
            max_concurrent: Number of concurrent processing tasks.

        Returns:
            Number of items successfully processed.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        success_count = 0

        async def limited_process(item):
            async with semaphore:
                return await self.process_item(item)

        tasks = [limited_process(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, bool) and res:
                success_count += 1
            elif isinstance(res, Exception):
                self.logger.error(f"Task failed with exception: {res}")

        self.logger.info(f"Preprocessing complete: {success_count}/{len(items)} items succeeded")
        return success_count

    # ------------------------------------------------------------------
    # Batch processing from MinIO or existing jobs
    # ------------------------------------------------------------------

    async def process_from_minio(self, prefix: str, limit: int = 1000) -> int:
        """
        Preprocess images already stored in MinIO (raw bucket) and save processed versions.
        """
        # List raw images
        objects = self.minio.list_objects(prefix, recursive=True)
        if not objects:
            self.logger.warning(f"No objects found under prefix: {prefix}")
            return 0

        items = []
        for obj in objects[:limit]:
            # Download raw image
            raw_data = self.minio.download_bytes(obj)
            if raw_data is None:
                continue
            items.append({"url": f"minio://{obj}", "metadata": {"raw_path": obj}})

        return await self.process_dataset(items)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_target_size(self, width: int, height: int) -> None:
        self.target_size = (width, height)

    def set_normalization(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std
        self.normalize = True

    def enable_augmentation(self, enable: bool = True) -> None:
        self.augment = enable

    def set_output_format(self, fmt: str) -> None:
        if fmt.lower() in ("jpg", "jpeg", "png", "npz"):
            self.output_format = fmt.lower()
        else:
            raise ValueError(f"Unsupported output format: {fmt}")