# ============================================================
# DOC AI Vision Service – Face Recognizer Engine
# ============================================================

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
from face_recognition import face_encodings, face_distance, compare_faces

from src.common.models import FaceResult

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition engine using dlib's face recognition model.

    Supports computing face encodings, comparing against known faces,
    and registering new faces. Encodings can be persisted to disk.
    """

    def __init__(self, known_faces_path: Optional[Path] = None):
        """
        Initialize the face recognizer.

        Args:
            known_faces_path: Path to a pickle file containing known face encodings.
                If not provided, starts with an empty database.
        """
        self.known_faces: Dict[str, List[np.ndarray]] = {}  # name -> list of encodings
        self.known_faces_path = known_faces_path
        self.tolerance = 0.6  # Standard face_recognition tolerance
        self.logger = logging.getLogger(f"{__name__}.FaceRecognizer")

        # Load known faces if path exists
        if known_faces_path and known_faces_path.exists():
            self._load_known_faces()

    def _load_known_faces(self) -> None:
        """Load known face encodings from a pickle file."""
        try:
            with open(self.known_faces_path, "rb") as f:
                self.known_faces = pickle.load(f)
            self.logger.info(f"Loaded {sum(len(v) for v in self.known_faces.values())} encodings for {len(self.known_faces)} people")
        except Exception as e:
            self.logger.error(f"Failed to load known faces from {self.known_faces_path}: {e}")
            self.known_faces = {}

    def _save_known_faces(self) -> None:
        """Save known face encodings to a pickle file."""
        if self.known_faces_path is None:
            self.logger.warning("No path set for known faces; not saving.")
            return
        try:
            with open(self.known_faces_path, "wb") as f:
                pickle.dump(self.known_faces, f)
            self.logger.info(f"Saved {sum(len(v) for v in self.known_faces.values())} encodings for {len(self.known_faces)} people")
        except Exception as e:
            self.logger.error(f"Failed to save known faces: {e}")

    def compute_encoding(self, face_image: np.ndarray) -> Optional[List[float]]:
        """
        Compute the face encoding for a cropped face image.

        Args:
            face_image: A numpy array representing the face (RGB or BGR).

        Returns:
            A list of 128 floats (encoding) if a face is detected, else None.
        """
        if face_image is None or face_image.size == 0:
            return None

        try:
            # face_recognition expects RGB, but we accept BGR and convert if needed
            if face_image.shape[2] == 3:
                # Check if it's BGR (OpenCV default) and convert to RGB
                # Simple heuristic: if the image comes from OpenCV, it's BGR.
                # We'll convert to RGB explicitly.
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image

            encodings = face_encodings(rgb_image)
            if encodings:
                return encodings[0].tolist()
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error computing face encoding: {e}")
            return None

    def recognize(self, face_image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Recognize a face by comparing its encoding to known faces.

        Args:
            face_image: A cropped face image (BGR or RGB).

        Returns:
            A tuple (name, confidence) where:
                name: The recognized person's name, or None if no match.
                confidence: A confidence score (1 - distance), or None if no match.
        """
        encoding = self.compute_encoding(face_image)
        if encoding is None:
            return None, None

        if not self.known_faces:
            self.logger.debug("No known faces to compare against.")
            return None, None

        # Flatten all known encodings with their corresponding names
        known_encodings = []
        known_names = []
        for name, enc_list in self.known_faces.items():
            for enc in enc_list:
                known_encodings.append(enc)
                known_names.append(name)

        if not known_encodings:
            return None, None

        # Compute distances
        distances = face_distance(known_encodings, encoding)
        if len(distances) == 0:
            return None, None

        min_distance = min(distances)
        if min_distance <= self.tolerance:
            best_index = int(np.argmin(distances))
            best_name = known_names[best_index]
            # Convert distance to confidence (0 to 1)
            confidence = 1.0 - min_distance / (self.tolerance * 2)  # heuristic
            confidence = max(0.0, min(1.0, confidence))
            self.logger.debug(f"Recognized {best_name} with confidence {confidence:.2f}")
            return best_name, confidence
        else:
            self.logger.debug("No matching face found")
            return None, None

    def register_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Register a new face by adding its encoding to the known database.

        Args:
            name: The person's name.
            face_image: A cropped face image.

        Returns:
            True if registration succeeded, False otherwise.
        """
        encoding = self.compute_encoding(face_image)
        if encoding is None:
            self.logger.warning(f"Could not compute encoding for {name}")
            return False

        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].append(np.array(encoding))
        self.logger.info(f"Registered face for {name} (total {len(self.known_faces[name])} encodings)")

        # Save to disk
        self._save_known_faces()
        return True

    def set_tolerance(self, tolerance: float) -> None:
        """Set the tolerance for face matching (lower = stricter)."""
        self.tolerance = max(0.0, min(1.0, tolerance))