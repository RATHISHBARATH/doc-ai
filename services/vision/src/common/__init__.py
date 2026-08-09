# ============================================================
# DOC AI Vision Service – Common Modules Package
# ============================================================

"""
Common utilities, clients, and configuration shared across all Vision components.
"""

from .config import load_config, get_config, reset_config, VisionConfig
from .minio_client import MinIOClient
from .postgres_client import PostgresClient
from .grpc_client import GRPCClient
from .models import (
    DetectionResult,
    OCRResult,
    FaceResult,
    Scene,
    VideoProcessingResult,
)

__all__ = [
    "load_config",
    "get_config",
    "reset_config",
    "VisionConfig",
    "MinIOClient",
    "PostgresClient",
    "GRPCClient",
    "DetectionResult",
    "OCRResult",
    "FaceResult",
    "Scene",
    "VideoProcessingResult",
]