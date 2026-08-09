# ============================================================
# DOC AI Vision Service – Video Processing Engine
# ============================================================

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from scenedetect import detect, ContentDetector, SceneManager
from scenedetect.video import VideoStream
from scenedetect.frame_timecode import FrameTimecode

from src.common.models import Scene, VideoProcessingResult

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing engine using OpenCV and scenedetect.

    Extracts frames at a configurable interval, segments video into scenes,
    and can run additional vision tasks on each frame.
    """

    def __init__(
        self,
        frame_interval: int = 1,
        scene_threshold: float = 30.0,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ):
        """
        Initialize the video processor.

        Args:
            frame_interval: Extract one frame every N seconds (default 1).
            scene_threshold: Scene detection sensitivity (higher = fewer scenes).
            target_fps: If set, resample video to this FPS.
            max_frames: Maximum number of frames to extract (0 = unlimited).
        """
        self.frame_interval = frame_interval
        self.scene_threshold = scene_threshold
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.logger = logging.getLogger(f"{__name__}.VideoProcessor")

    def process(
        self,
        video_path: Path,
        extract_frames: bool = True,
        segment_scenes: bool = True,
        run_vision_tasks: bool = False,
        vision_orchestrator = None,
    ) -> VideoProcessingResult:
        """
        Process a video file: extract frames, segment scenes, and optionally run vision tasks.

        Args:
            video_path: Path to the video file.
            extract_frames: Whether to extract and store individual frames.
            segment_scenes: Whether to detect scene boundaries.
            run_vision_tasks: Whether to run object detection, OCR, etc., on each frame.
                Requires vision_orchestrator to be provided.
            vision_orchestrator: An instance of VisionOrchestrator to run vision tasks.

        Returns:
            A VideoProcessingResult containing metadata, scenes, and optionally frame data.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Processing video: {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"Video: {width}x{height}, {fps:.2f} fps, {duration_seconds:.1f}s, {total_frames} frames")

        # Prepare result
        scenes = []
        frames_data = []  # List of (frame_number, timestamp, image_data)
        frame_detections = {}  # frame_number -> list of DetectionResult (if run_vision_tasks)
        frame_ocr = {}  # frame_number -> OCRResult
        frame_faces = {}  # frame_number -> list of FaceResult

        # Segment scenes if requested
        if segment_scenes:
            scene_boundaries = self._detect_scenes(str(video_path))
            self.logger.info(f"Detected {len(scene_boundaries)} scenes")
            # Create Scene objects
            for i, (start, end) in enumerate(scene_boundaries):
                start_frame = int(start.get_frames())
                end_frame = int(end.get_frames())
                # Store frame numbers; description can be added later
                scenes.append(Scene(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    description=f"Scene {i+1}",
                    confidence=None,  # scenedetect doesn't provide confidence
                ))

        # Extract frames if requested
        if extract_frames:
            frames_data = self._extract_frames(cap, total_frames)
            self.logger.info(f"Extracted {len(frames_data)} frames")

        # Run vision tasks on extracted frames (if enabled)
        if run_vision_tasks and vision_orchestrator is not None:
            self.logger.info("Running vision tasks on extracted frames...")
            for frame_num, timestamp, img_data in frames_data:
                # Decode image from bytes to numpy array
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Run selected tasks (e.g., object detection, OCR, face)
                # For simplicity, we run all available tasks; could be configurable.
                detections = vision_orchestrator.yolo.detect(img)
                ocr_result = vision_orchestrator.tesseract.ocr(img)
                faces = vision_orchestrator.face_detector.detect(img)

                if detections:
                    frame_detections[frame_num] = detections
                if ocr_result and ocr_result.text:
                    frame_ocr[frame_num] = ocr_result
                if faces:
                    frame_faces[frame_num] = faces

            self.logger.info("Vision tasks completed on extracted frames.")

        # Release capture
        cap.release()

        # Build result
        result = VideoProcessingResult(
            video_url=str(video_path),
            duration_seconds=duration_seconds,
            total_frames=total_frames,
            scenes=scenes,
            detections=[d for det_list in frame_detections.values() for d in det_list],
            faces=[f for face_list in frame_faces.values() for f in face_list],
            ocr_text="\n".join([f"Frame {fn}: {ocr.text}" for fn, ocr in frame_ocr.items()]),
            pose_results=[],  # Can be added later
            metadata={
                "width": width,
                "height": height,
                "fps": fps,
                "frame_interval": self.frame_interval,
                "extracted_frames": len(frames_data),
                "scenes_detected": len(scenes),
            }
        )

        self.logger.info(f"Video processing complete. Duration: {duration_seconds:.1f}s, scenes: {len(scenes)}")
        return result

    def _detect_scenes(self, video_path: str) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
        Detect scene boundaries using scenedetect.

        Returns:
            A list of (start_timecode, end_timecode) tuples.
        """
        try:
            video_stream = VideoStream(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
            scene_manager.detect_scenes(video_stream)

            # Get scene list
            scene_list = scene_manager.get_scene_list()
            if not scene_list:
                # If no scenes detected, treat entire video as one scene
                video_stream = VideoStream(video_path)
                duration = video_stream.duration
                return [(FrameTimecode(0, fps=video_stream.fps), duration)]

            return scene_list
        except Exception as e:
            self.logger.error(f"Scene detection failed: {e}")
            return []

    def _extract_frames(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
    ) -> List[Tuple[int, float, bytes]]:
        """
        Extract frames from the video at the configured interval.

        Returns:
            A list of (frame_number, timestamp_seconds, image_data_bytes) tuples.
        """
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # fallback

        # Determine frame interval in frames (not seconds)
        interval_frames = int(self.frame_interval * fps)
        if interval_frames < 1:
            interval_frames = 1

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
                # Encode frame as JPEG bytes
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buffer.tobytes()
                timestamp = frame_count / fps
                frames.append((frame_count, timestamp, img_bytes))
                extracted_count += 1

                # Stop if we've reached max_frames
                if self.max_frames and extracted_count >= self.max_frames:
                    break

            frame_count += 1

        return frames

    def extract_frames_from_path(self, video_path: Path) -> List[Tuple[int, float, bytes]]:
        """
        Convenience method to extract frames from a video file path.

        Returns:
            List of (frame_number, timestamp, image_data_bytes).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result = self._extract_frames(cap, total_frames)
        cap.release()
        return result

    def get_metadata(self, video_path: Path) -> Dict[str, Any]:
        """
        Get metadata from a video file without extracting frames.

        Returns:
            Dictionary with width, height, fps, total_frames, duration_seconds.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration_seconds,
        }

    def close(self) -> None:
        """Release any resources (none currently)."""
        pass