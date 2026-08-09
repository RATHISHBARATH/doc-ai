# ============================================================
# DOC AI Vision Service – Package Initialization
# ============================================================

"""
Vision Intelligence Service

This package provides computer vision capabilities including:
- Object detection (YOLOv8)
- OCR (Tesseract)
- Face detection and recognition (dlib)
- Pose estimation and gesture recognition (MediaPipe)
- Video processing (frame extraction, scene segmentation)
- Automated web crawling and dataset ingestion
- Storage and retrieval of visual metadata in MinIO and Milvus
"""

__version__ = "0.1.0"
__author__ = "SpyWeb AI Labs"

# Core modules
from . import api
from . import core
from . import engines
from . import model_loader
from . import crawler
from . import dataset_ingestion
from . import validation
from . import common

# Expose key functions for external use
from .core.orchestrator import VisionOrchestrator
from .common.config import get_config, VisionConfig

__all__ = [
    "api",
    "core",
    "engines",
    "model_loader",
    "crawler",
    "dataset_ingestion",
    "validation",
    "common",
    "VisionOrchestrator",
    "get_config",
    "VisionConfig",
]