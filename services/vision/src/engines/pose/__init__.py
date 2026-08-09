# ============================================================
# DOC AI Vision Service – Pose Estimation Engine Package
# ============================================================

"""
Pose estimation and gesture recognition engine using MediaPipe.

This package provides the MediaPipePose class, which detects human pose
landmarks and can estimate gestures from images and video frames.
"""

from .mediapipe_pose import MediaPipePose

__all__ = [
    "MediaPipePose",
]