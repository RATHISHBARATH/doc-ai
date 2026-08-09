# ============================================================
# DOC AI Vision Service – gRPC Server (Corrected with entry point)
# ============================================================

import asyncio
import logging
import grpc
import cv2
import numpy as np

from src.common.config import get_config
from src.common.models import VisionTaskType
from src.core.orchestrator import VisionOrchestrator

# Import generated protobuf stubs from the shared doc_ai_common package
from doc_ai_common import vision_pb2
from doc_ai_common import vision_pb2_grpc

logger = logging.getLogger(__name__)


class VisionServicer(vision_pb2_grpc.VisionServicer):
    """
    gRPC service implementation for the Vision API.
    """

    def __init__(self, orchestrator: VisionOrchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{__name__}.VisionServicer")

    async def DetectObjects(self, request, context):
        """
        Detect objects in an image.
        """
        try:
            # Process the image
            result = await self.orchestrator.process_image(
                image_data=request.image_data,
                tasks=[VisionTaskType.OBJECT_DETECTION],
                store_result=True,
            )
            detections = result.get("detections", [])
            # Build response
            response = vision_pb2.DetectResponse()
            for d in detections:
                obj = response.objects.add()
                obj.label = d["label"]
                obj.confidence = d["confidence"]
                obj.x = d["x"]
                obj.y = d["y"]
                obj.width = d["width"]
                obj.height = d["height"]
            return response
        except Exception as e:
            self.logger.error(f"DetectObjects error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.DetectResponse()

    async def OCR(self, request, context):
        """
        Extract text from an image.
        """
        try:
            result = await self.orchestrator.process_image(
                image_data=request.image_data,
                tasks=[VisionTaskType.OCR],
                store_result=True,
            )
            ocr = result.get("ocr", {})
            response = vision_pb2.OCRResponse()
            response.text = ocr.get("text", "")
            return response
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.OCRResponse()

    async def RecognizeFaces(self, request, context):
        """
        Recognize faces in an image.
        """
        try:
            result = await self.orchestrator.process_image(
                image_data=request.image_data,
                tasks=[VisionTaskType.FACE_RECOGNITION],
                store_result=True,
            )
            faces = result.get("faces", [])
            response = vision_pb2.FaceResponse()
            for f in faces:
                face = response.faces.add()
                face.x = f["x"]
                face.y = f["y"]
                face.width = f["width"]
                face.height = f["height"]
                face.name = f.get("name", "")
                face.confidence = f.get("confidence", 0.0)
            return response
        except Exception as e:
            self.logger.error(f"RecognizeFaces error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.FaceResponse()

    async def ProcessVideo(self, request, context):
        """
        Process a video: extract frames, segment scenes, and run vision tasks.
        """
        try:
            # Note: video_url is expected to be a path or a MinIO URI.
            # For simplicity, we assume it's a local path or a downloadable URL.
            result = await self.orchestrator.process_video(
                video_url=request.video_url,
                extract_frames=request.extract_frames,
                segment_scenes=request.segment_scenes,
                store_result=True,
            )
            response = vision_pb2.VideoResponse()
            response.frame_count = result.get("total_frames", 0)
            for scene in result.get("scenes", []):
                s = response.scenes.add()
                s.start_frame = scene["start_frame"]
                s.end_frame = scene["end_frame"]
                s.description = scene.get("description", "")
                # Embedding and video_url can be added as needed
            return response
        except Exception as e:
            self.logger.error(f"ProcessVideo error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.VideoResponse()

    async def QueryScenes(self, request, context):
        """
        Query scenes by semantic embedding.
        """
        try:
            # Placeholder – implement Milvus query later.
            response = vision_pb2.QueryResponse()
            # Not yet implemented
            self.logger.warning("QueryScenes not yet fully implemented")
            return response
        except Exception as e:
            self.logger.error(f"QueryScenes error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vision_pb2.QueryResponse()


async def serve_grpc(orchestrator: VisionOrchestrator, port: int = 50055) -> None:
    """
    Start the gRPC server.
    """
    server = grpc.aio.server()
    vision_pb2_grpc.add_VisionServicer_to_server(
        VisionServicer(orchestrator), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


def run_sync_grpc(orchestrator: VisionOrchestrator, port: int = 50055) -> None:
    """
    Synchronous entry point for the gRPC server (for use with uvicorn/threads).
    """
    asyncio.run(serve_grpc(orchestrator, port))


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Load configuration and create orchestrator
    config = get_config()
    orchestrator = VisionOrchestrator()
    port = config.grpc_port
    logger.info(f"Starting Vision gRPC server on port {port}")
    run_sync_grpc(orchestrator, port)