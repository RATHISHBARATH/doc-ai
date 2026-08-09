# ============================================================
# DOC AI Vision Service – MediaPipe Pose Engine
# ============================================================

import logging
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
import mediapipe as mp

from src.common.models import PoseResult

logger = logging.getLogger(__name__)


class MediaPipePose:
    """
    Pose estimation engine using MediaPipe Pose.

    Detects human pose landmarks (33 keypoints) in images and video frames.
    Can also estimate basic gestures (e.g., raising a hand).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the MediaPipe Pose model.

        Args:
            static_image_mode: If True, runs detection on every frame (slower).
            model_complexity: 0 (lite), 1 (full), or 2 (heavy).
            enable_segmentation: Whether to output segmentation masks.
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.logger = logging.getLogger(f"{__name__}.MediaPipePose")

    def estimate(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose from an image.

        Args:
            image: A numpy array representing the image (BGR format, OpenCV style).

        Returns:
            A PoseResult object containing landmarks and score, or None if no pose detected.
        """
        if image is None:
            self.logger.warning("Received None image, returning None")
            return None

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = self.pose.process(rgb_image)

            if not results.pose_landmarks:
                self.logger.debug("No pose detected")
                return None

            # Extract landmarks as a list of dictionaries
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

            # Calculate a confidence score based on average visibility
            visibility_scores = [lm.visibility for lm in results.pose_landmarks.landmark]
            avg_score = sum(visibility_scores) / len(visibility_scores) if visibility_scores else None

            return PoseResult(
                landmarks=landmarks,
                score=avg_score,
                handedness=None,  # Pose doesn't provide handedness; only hand landmarks do
            )

        except Exception as e:
            self.logger.error(f"Pose estimation error: {e}")
            return None

    def estimate_batch(self, images: List[np.ndarray]) -> List[Optional[PoseResult]]:
        """
        Estimate pose for multiple images.

        Args:
            images: List of images (BGR arrays).

        Returns:
            A list of PoseResult objects (or None for each image).
        """
        results = []
        for img in images:
            results.append(self.estimate(img))
        return results

    def draw_landmarks(self, image: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        """
        Draw pose landmarks on a copy of the image.

        Args:
            image: Original BGR image.
            pose_result: The PoseResult containing landmarks.

        Returns:
            A copy of the image with landmarks drawn.
        """
        if pose_result is None or pose_result.landmarks is None:
            return image

        # Convert landmarks back to MediaPipe NormalizedLandmarkList format
        landmark_list = self.mp_pose.NormalizedLandmarkList()
        for lm_dict in pose_result.landmarks:
            lm = landmark_list.landmark.add()
            lm.x = lm_dict["x"]
            lm.y = lm_dict["y"]
            lm.z = lm_dict.get("z", 0.0)
            lm.visibility = lm_dict.get("visibility", 1.0)

        # Draw on a copy of the image
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
        )
        return annotated_image

    def detect_gesture(self, pose_result: PoseResult) -> Optional[str]:
        """
        Detect a simple gesture from the pose landmarks.

        Args:
            pose_result: The PoseResult containing landmarks.

        Returns:
            A gesture name (e.g., "left_hand_up", "right_hand_up", "both_hands_up"),
            or None if no recognized gesture.
        """
        if pose_result is None or pose_result.landmarks is None:
            return None

        # The landmark indices we need (MediaPipe Pose landmarks)
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26

        # Get landmarks
        lm = {i: pose_result.landmarks[i] for i in range(len(pose_result.landmarks))}

        # Check if hands are raised above shoulders
        left_hand_raised = False
        right_hand_raised = False

        if LEFT_WRIST in lm and LEFT_SHOULDER in lm:
            if lm[LEFT_WRIST]["y"] < lm[LEFT_SHOULDER]["y"]:
                left_hand_raised = True

        if RIGHT_WRIST in lm and RIGHT_SHOULDER in lm:
            if lm[RIGHT_WRIST]["y"] < lm[RIGHT_SHOULDER]["y"]:
                right_hand_raised = True

        if left_hand_raised and right_hand_raised:
            return "both_hands_up"
        elif left_hand_raised:
            return "left_hand_up"
        elif right_hand_raised:
            return "right_hand_up"

        # More gestures can be added here (e.g., waving, pointing)
        return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
        self.logger.info("MediaPipe Pose resources released")