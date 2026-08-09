# ============================================================
# DOC AI Vision Service – Face Engine Package
# ============================================================

"""
Face detection and recognition engines using dlib and face_recognition.

This package provides:
- FaceDetector: Detects faces in images using dlib's HOG or CNN detector.
- FaceRecognizer: Recognizes faces using dlib's face recognition model.
"""

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
]