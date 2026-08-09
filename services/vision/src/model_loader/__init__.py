# ============================================================
# DOC AI Vision Service – Model Loader Package
# ============================================================

"""
Model loading and caching for vision engines.

This package provides a centralized ModelLoader that loads and caches
pre-trained models (YOLO, Tesseract, dlib, MediaPipe) to avoid redundant
loading across multiple engine instances.
"""

from .loader import ModelLoader

__all__ = [
    "ModelLoader",
]