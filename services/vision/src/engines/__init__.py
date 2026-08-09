# ============================================================
# DOC AI Vision Service – Engines Package
# ============================================================

"""
Vision processing engines.

This package contains the specialized processing modules:
- Object detection (YOLO)
- OCR (Tesseract)
- Face detection and recognition (dlib)
- Pose estimation (MediaPipe)
- Video processing (frame extraction, scene segmentation)  <-- temporarily disabled
"""

from .object_detection.yolo import YOLODetector
from .ocr.tesseract import TesseractOCR
from .face.face_detector import FaceDetector
from .face.face_recognizer import FaceRecognizer
from .pose.mediapipe_pose import MediaPipePose
# from .video import VideoProcessor   # ← commented out

__all__ = [
    "YOLODetector",
    "TesseractOCR",
    "FaceDetector",
    "FaceRecognizer",
    "MediaPipePose",
    # "VideoProcessor",               # ← commented out
]