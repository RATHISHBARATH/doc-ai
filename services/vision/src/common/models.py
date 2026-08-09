# ============================================================
# DOC AI Vision Service – Data Models
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class VisionTaskType(str, Enum):
    """Types of vision tasks that can be performed."""
    OBJECT_DETECTION = "object_detection"
    OCR = "ocr"
    FACE_RECOGNITION = "face_recognition"
    POSE_ESTIMATION = "pose_estimation"
    SCENE_SEGMENTATION = "scene_segmentation"
    VIDEO_ANALYSIS = "video_analysis"


@dataclass
class DetectionResult:
    """Result from object detection (YOLO)."""
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
        )


@dataclass
class OCRResult:
    """Result from OCR (Tesseract)."""
    text: str
    confidence: Optional[float] = None
    bounding_box: Optional[List[float]] = None  # [x, y, width, height]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        return cls(
            text=data["text"],
            confidence=data.get("confidence"),
            bounding_box=data.get("bounding_box"),
        )


@dataclass
class FaceResult:
    """Result from face detection/recognition."""
    x: float
    y: float
    width: float
    height: float
    name: Optional[str] = None  # Recognized name, if known
    confidence: Optional[float] = None
    landmarks: Optional[List[Dict[str, float]]] = None  # Eye, nose, mouth positions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "name": self.name,
            "confidence": self.confidence,
            "landmarks": self.landmarks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceResult":
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            name=data.get("name"),
            confidence=data.get("confidence"),
            landmarks=data.get("landmarks"),
        )


@dataclass
class PoseResult:
    """Result from pose estimation (MediaPipe)."""
    landmarks: List[Dict[str, float]]  # Body landmarks (x, y, z, visibility)
    score: Optional[float] = None
    handedness: Optional[str] = None  # "left" or "right" for hand landmarks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "landmarks": self.landmarks,
            "score": self.score,
            "handedness": self.handedness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseResult":
        return cls(
            landmarks=data["landmarks"],
            score=data.get("score"),
            handedness=data.get("handedness"),
        )


@dataclass
class Scene:
    """A video scene segment."""
    start_frame: int
    end_frame: int
    description: Optional[str] = None
    embedding: Optional[List[float]] = None  # Vector embedding for retrieval
    video_url: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "description": self.description,
            "embedding": self.embedding,
            "video_url": self.video_url,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        return cls(
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            description=data.get("description"),
            embedding=data.get("embedding"),
            video_url=data.get("video_url"),
            confidence=data.get("confidence"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VideoProcessingResult:
    """Aggregated result from processing a video."""
    video_url: str
    duration_seconds: float
    total_frames: int
    scenes: List[Scene] = field(default_factory=list)
    detections: List[DetectionResult] = field(default_factory=list)
    faces: List[FaceResult] = field(default_factory=list)
    ocr_text: Optional[str] = None
    pose_results: List[PoseResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_url": self.video_url,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "scenes": [s.to_dict() for s in self.scenes],
            "detections": [d.to_dict() for d in self.detections],
            "faces": [f.to_dict() for f in self.faces],
            "ocr_text": self.ocr_text,
            "pose_results": [p.to_dict() for p in self.pose_results],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoProcessingResult":
        return cls(
            video_url=data["video_url"],
            duration_seconds=data["duration_seconds"],
            total_frames=data["total_frames"],
            scenes=[Scene.from_dict(s) for s in data.get("scenes", [])],
            detections=[DetectionResult.from_dict(d) for d in data.get("detections", [])],
            faces=[FaceResult.from_dict(f) for f in data.get("faces", [])],
            ocr_text=data.get("ocr_text"),
            pose_results=[PoseResult.from_dict(p) for p in data.get("pose_results", [])],
            metadata=data.get("metadata", {}),
        )