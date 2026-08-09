# ============================================================
# DOC AI Vision Service – Model Loader
# ============================================================

import logging
from typing import Optional, Dict, Any

import dlib
import mediapipe as mp
import pytesseract
from ultralytics import YOLO

from src.common.config import get_config, VisionConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Centralized model loader and cache for vision engines.

    This class ensures that heavy models (YOLO, dlib, MediaPipe, Tesseract)
    are loaded only once and reused across the service. It also provides
    a single point for future model versioning, downloading, and swapping.
    """

    _instance: Optional["ModelLoader"] = None

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.config: VisionConfig = get_config()
            self.logger = logging.getLogger(f"{__name__}.ModelLoader")

            # Cache for loaded models
            self._yolo: Optional[YOLO] = None
            self._face_detector: Optional[dlib.fhog_object_detector] = None  # HOG detector
            self._cnn_face_detector: Optional[dlib.cnn_face_detection_model_v1] = None
            self._mediapipe_pose: Optional[mp.solutions.pose.Pose] = None

            # Tesseract is a system-level engine; we just set its path/config via environment
            self._tesseract_configured: bool = False

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def get_yolo(self, model_path: Optional[str] = None) -> YOLO:
        """
        Load and cache the YOLO model.

        Args:
            model_path: Optional override for the model path.
                        If not provided, uses config.models.object_detection.

        Returns:
            The loaded YOLO model.
        """
        if self._yolo is not None:
            return self._yolo

        model_path = model_path or self.config.models.object_detection
        try:
            self._yolo = YOLO(model_path)
            self.logger.info(f"YOLO model loaded: {model_path}")
            return self._yolo
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise

    # ------------------------------------------------------------------
    # Face Detector (dlib)
    # ------------------------------------------------------------------

    def get_face_detector(self, method: str = "hog") -> Any:
        """
        Load and cache the dlib face detector (HOG or CNN).

        Args:
            method: 'hog' or 'cnn'.

        Returns:
            The loaded detector (dlib.fhog_object_detector or cnn_face_detection_model_v1).
        """
        if method == "hog":
            if self._face_detector is None:
                try:
                    self._face_detector = dlib.get_frontal_face_detector()
                    self.logger.info("dlib HOG face detector loaded")
                except Exception as e:
                    self.logger.error(f"Failed to load HOG face detector: {e}")
                    raise
            return self._face_detector

        elif method == "cnn":
            if self._cnn_face_detector is None:
                # CNN model requires a model file; default location can be overridden
                model_path = "/app/models/mmod_human_face_detector.dat"
                try:
                    self._cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)
                    self.logger.info(f"dlib CNN face detector loaded from {model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load CNN face detector from {model_path}: {e}")
                    raise
            return self._cnn_face_detector

        else:
            raise ValueError(f"Invalid face detector method: {method}")

    # ------------------------------------------------------------------
    # MediaPipe Pose
    # ------------------------------------------------------------------

    def get_mediapipe_pose(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> mp.solutions.pose.Pose:
        """
        Load and cache the MediaPipe Pose model.

        All arguments are passed directly to the MediaPipe constructor.
        Note: If the model is already loaded with different parameters,
        this method will recreate it with the new parameters.
        """
        if self._mediapipe_pose is not None:
            # For simplicity, we reuse the existing instance regardless of parameters.
            # In a production system, you might want to check parameters and reload if needed.
            return self._mediapipe_pose

        try:
            mp_pose = mp.solutions.pose
            self._mediapipe_pose = mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                enable_segmentation=enable_segmentation,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.logger.info("MediaPipe Pose model loaded")
            return self._mediapipe_pose
        except Exception as e:
            self.logger.error(f"Failed to load MediaPipe Pose: {e}")
            raise

    # ------------------------------------------------------------------
    # Tesseract (configure once)
    # ------------------------------------------------------------------

    def configure_tesseract(self, tesseract_path: Optional[str] = None) -> None:
        """
        Configure the Tesseract OCR engine.

        This sets the path to the Tesseract executable and optionally
        configures language packs. This should be called once at startup.
        """
        if self._tesseract_configured:
            return

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.logger.info(f"Tesseract path set to: {tesseract_path}")
        else:
            # Assume it's in PATH (default on most systems)
            self.logger.info("Tesseract using default system PATH")

        # Optionally set TESSDATA_PREFIX if needed
        # (can be set via environment variable)

        self._tesseract_configured = True

    # ------------------------------------------------------------------
    # Utility / cleanup
    # ------------------------------------------------------------------

    def reload_all(self) -> None:
        """
        Force reload all models (clear cache).
        Use this when models have been updated or replaced.
        """
        self._yolo = None
        self._face_detector = None
        self._cnn_face_detector = None
        self._mediapipe_pose = None
        self._tesseract_configured = False
        self.logger.info("All model caches cleared (will reload on next request)")

    def close(self) -> None:
        """Release any resources held by the loaded models."""
        if self._mediapipe_pose is not None:
            self._mediapipe_pose.close()
            self._mediapipe_pose = None
            self.logger.info("MediaPipe Pose resources released")
        # YOLO and dlib do not require explicit cleanup.
        self.reload_all()