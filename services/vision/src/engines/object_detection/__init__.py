# ============================================================
# DOC AI Vision Service – Object Detection Engine Package
# ============================================================

"""
Object detection engine using YOLOv8.

This module provides the YOLODetector class, which wraps Ultralytics YOLO
for object detection tasks.
"""

from .yolo import YOLODetector

__all__ = [
    "YOLODetector",
]