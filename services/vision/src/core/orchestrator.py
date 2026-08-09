# ============================================================
# DOC AI Vision Service – Core Orchestrator
# ============================================================

import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from src.common.config import get_config, VisionConfig
from src.common.models import (
    VisionTaskType,
    DetectionResult,
    OCRResult,
    FaceResult,
    PoseResult,
    Scene,
    VideoProcessingResult,
)
from src.common.minio_client import MinIOClient
from src.common.postgres_client import PostgresClient
from src.engines.object_detection.yolo import YOLODetector
from src.engines.face.face_detector import FaceDetector
from src.engines.face.face_recognizer import FaceRecognizer
from src.engines.ocr.tesseract import TesseractOCR
from src.engines.pose.mediapipe_pose import MediaPipePose
# from src.engines.video import VideoProcessor   # ← Video engine disabled

logger = logging.getLogger(__name__)


class VisionOrchestrator:
    """
    Central orchestrator for all vision tasks.
    
    Coordinates image and video processing by:
    - Managing engine instances and their lifecycle.
    - Routing requests to the appropriate engine(s).
    - Aggregating results into structured responses.
    - Persisting metadata to PostgreSQL and files to MinIO.
    """

    def __init__(self):
        self.config: VisionConfig = get_config()
        self.minio = MinIOClient(self.config)
        self.postgres = PostgresClient()
        
        # Initialize engines (lazy-loaded on first use)
        self._yolo: Optional[YOLODetector] = None
        self._tesseract: Optional[TesseractOCR] = None
        self._face_detector: Optional[FaceDetector] = None
        self._face_recognizer: Optional[FaceRecognizer] = None
        self._pose: Optional[MediaPipePose] = None
        self._video: Optional[VideoProcessor] = None  # Will remain None
        
        self.logger = logging.getLogger(f"{__name__}.VisionOrchestrator")

    # ------------------------------------------------------------------
    # Engine lazy-loading
    # ------------------------------------------------------------------

    @property
    def yolo(self) -> YOLODetector:
        if self._yolo is None:
            self._yolo = YOLODetector(self.config.models.object_detection)
        return self._yolo

    @property
    def tesseract(self) -> TesseractOCR:
        if self._tesseract is None:
            self._tesseract = TesseractOCR()
        return self._tesseract

    @property
    def face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        return self._face_detector

    @property
    def face_recognizer(self) -> FaceRecognizer:
        if self._face_recognizer is None:
            self._face_recognizer = FaceRecognizer()
        return self._face_recognizer

    @property
    def pose(self) -> MediaPipePose:
        if self._pose is None:
            self._pose = MediaPipePose()
        return self._pose

    # @property
    # def video(self) -> VideoProcessor:
    #     if self._video is None:
    #         self._video = VideoProcessor(
    #             frame_interval=self.config.video.frame_interval,
    #             scene_threshold=self.config.video.scene_threshold,
    #         )
    #     return self._video

    # ------------------------------------------------------------------
    # Image processing methods
    # ------------------------------------------------------------------

    async def process_image(
        self,
        image_data: bytes,
        tasks: List[VisionTaskType],
        store_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an image with the specified vision tasks.

        Args:
            image_data: Raw image bytes (JPEG, PNG, etc.).
            tasks: List of vision tasks to perform.
            store_result: Whether to store results in MinIO/PostgreSQL.

        Returns:
            A dictionary containing the results for each task.
        """
        result = {}
        job_id = None
        
        if store_result:
            job_id = str(uuid.uuid4())
            # Store raw image in MinIO
            remote_path = f"{self.config.storage.raw_images_prefix}{job_id}.jpg"
            self.minio.upload_bytes(image_data, remote_path, "image/jpeg")
            self.postgres.insert_job(job_id, remote_path, "image")

        # Decode image for OpenCV processing
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if VisionTaskType.OBJECT_DETECTION in tasks:
            detections = self.yolo.detect(img)
            result["detections"] = [d.to_dict() for d in detections]
            if store_result and job_id:
                for d in detections:
                    self.postgres.insert_detection(
                        job_id=job_id,
                        frame_number=0,
                        label=d.label,
                        confidence=d.confidence,
                        x=d.x,
                        y=d.y,
                        width=d.width,
                        height=d.height,
                    )

        if VisionTaskType.OCR in tasks:
            ocr_result = self.tesseract.ocr(img)
            result["ocr"] = ocr_result.to_dict()
            if store_result and job_id:
                self.postgres.insert_ocr_result(
                    job_id=job_id,
                    frame_number=0,
                    text=ocr_result.text,
                    confidence=ocr_result.confidence,
                    bounding_box=ocr_result.bounding_box,
                )

        if VisionTaskType.FACE_RECOGNITION in tasks:
            faces = self.face_detector.detect(img)
            # Optionally recognize faces
            recognized = []
            for face in faces:
                face_img = img[face.y:face.y+face.height, face.x:face.x+face.width]
                name, conf = self.face_recognizer.recognize(face_img)
                if name:
                    face.name = name
                    face.confidence = conf
                recognized.append(face)
            result["faces"] = [f.to_dict() for f in recognized]
            if store_result and job_id:
                for f in recognized:
                    self.postgres.insert_face(
                        job_id=job_id,
                        frame_number=0,
                        x=f.x,
                        y=f.y,
                        width=f.width,
                        height=f.height,
                        name=f.name,
                        confidence=f.confidence,
                        landmarks=f.landmarks,
                    )

        if VisionTaskType.POSE_ESTIMATION in tasks:
            pose_result = self.pose.estimate(img)
            result["pose"] = pose_result.to_dict() if pose_result else None

        if store_result and job_id:
            self.postgres.complete_job(job_id)

        return result

    # ------------------------------------------------------------------
    # Video processing methods (temporarily disabled)
    # ------------------------------------------------------------------

    async def process_video(
        self,
        video_url: str,
        extract_frames: bool = True,
        segment_scenes: bool = True,
        store_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Video processing is currently disabled due to a dependency issue.
        This stub returns a message and an empty result.
        """
        self.logger.warning("Video processing is disabled. Returning empty result.")
        return {
            "status": "disabled",
            "message": "Video processing is not available in this build.",
            "video_url": video_url,
        }

    # ------------------------------------------------------------------
    # Combined / multimodal processing
    # ------------------------------------------------------------------

    async def process_multimodal(
        self,
        image_data: Optional[bytes] = None,
        video_url: Optional[str] = None,
        tasks: List[VisionTaskType] = None,
    ) -> Dict[str, Any]:
        """
        Process both image and video inputs together (multimodal).

        Args:
            image_data: Raw image bytes (optional).
            video_url: Video URL (optional).
            tasks: List of vision tasks to perform.

        Returns:
            Combined results.
        """
        result = {}
        if image_data:
            result["image"] = await self.process_image(image_data, tasks, store_result=True)
        if video_url:
            result["video"] = await self.process_video(video_url, store_result=True)
        return result