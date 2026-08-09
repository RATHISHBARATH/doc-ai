# ============================================================
# DOC AI Vision Service – YOLO Object Detection Engine
# ============================================================

import logging
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from src.common.models import DetectionResult

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Object detection engine using YOLOv8.

    This class loads a YOLO model (e.g., yolov8n.pt) and provides
    a `detect` method that returns a list of `DetectionResult` objects.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO model file or name (e.g., 'yolov8n.pt').
                       Supports Ultralytics built-in models.
        """
        self.model_path = model_path
        self.confidence_threshold = 0.25   # adjustable
        self.iou_threshold = 0.45          # adjustable
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in an image.

        Args:
            image: A numpy array representing the image (BGR format, OpenCV style).

        Returns:
            A list of DetectionResult objects, each representing a detected object.
            If no objects are detected, an empty list is returned.
        """
        if self.model is None:
            logger.error("YOLO model not loaded. Cannot run detection.")
            return []

        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # Parse results
            detections = []
            for result in results:
                for box in result.boxes:
                    # Extract coordinates (x1, y1, x2, y2) and confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id] if self.model.names else str(cls_id)

                    # Convert to DetectionResult (x, y, width, height)
                    x = x1
                    y = y1
                    width = x2 - x1
                    height = y2 - y1

                    detections.append(DetectionResult(
                        label=label,
                        confidence=conf,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    ))

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            return []

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for detections (0.0 to 1.0)."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def set_iou_threshold(self, threshold: float) -> None:
        """Set the IoU threshold for non‑maximum suppression (0.0 to 1.0)."""
        self.iou_threshold = max(0.0, min(1.0, threshold))