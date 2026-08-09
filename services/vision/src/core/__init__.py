# ============================================================
# DOC AI Vision Service – Core Package
# ============================================================

"""
Core orchestration logic for the Vision service.

The core module is responsible for:
- Receiving vision tasks from the API layer.
- Coordinating the appropriate engines (object detection, OCR, face, pose, video).
- Aggregating results into structured responses.
- Managing job lifecycle and storage.
"""

from .orchestrator import VisionOrchestrator

__all__ = [
    "VisionOrchestrator",
]