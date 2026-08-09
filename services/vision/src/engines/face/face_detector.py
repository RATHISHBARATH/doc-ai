# ============================================================
# DOC AI Vision Service – Face Detector Engine
# ============================================================

import logging
from typing import List, Optional

import cv2
import dlib
import numpy as np

from src.common.models import FaceResult

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detection engine using dlib.

    Supports both HOG (fast, CPU) and CNN (more accurate, GPU) detectors.
    """

    def __init__(self, method: str = "hog", upscale: int = 1):
        """
        Initialize the face detector.

        Args:
            method: 'hog' or 'cnn'. Default is 'hog'.
            upscale: Number of times to upscale the image (1 = no upscale).
        """
        self.method = method.lower()
        self.upscale = upscale
        self.detector = None
        self._load_detector()
        self.logger = logging.getLogger(f"{__name__}.FaceDetector")

    def _load_detector(self) -> None:
        """Load the appropriate dlib detector based on the method."""
        try:
            if self.method == "hog":
                self.detector = dlib.get_frontal_face_detector()
                self.logger.info("Loaded dlib HOG face detector")
            elif self.method == "cnn":
                # Load the CNN model (requires the model file)
                # For production, you should download the model and place it in a known path.
                model_path = "/app/models/mmod_human_face_detector.dat"
                try:
                    self.detector = dlib.cnn_face_detection_model_v1(model_path)
                    self.logger.info("Loaded dlib CNN face detector")
                except Exception as e:
                    self.logger.warning(
                        f"CNN model not found at {model_path}, falling back to HOG: {e}"
                    )
                    self.detector = dlib.get_frontal_face_detector()
                    self.method = "hog"
            else:
                raise ValueError(f"Invalid face detector method: {self.method}")
        except Exception as e:
            self.logger.error(f"Failed to load face detector: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[FaceResult]:
        """
        Detect faces in an image.

        Args:
            image: A numpy array representing the image (BGR format, OpenCV style).

        Returns:
            A list of FaceResult objects, each containing the bounding box.
            If no faces are detected, an empty list is returned.
        """
        if image is None:
            self.logger.warning("Received None image, returning empty face list")
            return []

        try:
            # Convert BGR to RGB (dlib expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces
            if self.method == "hog":
                faces = self.detector(rgb_image, self.upscale)
                # dlib returns rectangles; we convert them to FaceResult
                results = []
                for rect in faces:
                    results.append(
                        FaceResult(
                            x=float(rect.left()),
                            y=float(rect.top()),
                            width=float(rect.width()),
                            height=float(rect.height()),
                        )
                    )
            elif self.method == "cnn":
                # CNN detector returns mmod_rectangles with confidence
                detections = self.detector(rgb_image, self.upscale)
                results = []
                for detection in detections:
                    rect = detection.rect
                    confidence = detection.confidence
                    results.append(
                        FaceResult(
                            x=float(rect.left()),
                            y=float(rect.top()),
                            width=float(rect.width()),
                            height=float(rect.height()),
                            confidence=float(confidence),
                        )
                    )
            else:
                # Should not happen
                results = []

            self.logger.debug(f"Detected {len(results)} faces")
            return results

        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return []

    def set_method(self, method: str) -> None:
        """Change the detection method (reloads the detector)."""
        if method.lower() != self.method:
            self.method = method.lower()
            self._load_detector()