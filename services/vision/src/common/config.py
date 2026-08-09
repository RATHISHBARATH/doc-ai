# ============================================================
# DOC AI Vision Service – Configuration Loader
# ============================================================

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelsConfig:
    """Configuration for vision models."""
    object_detection: str = "yolov8n.pt"
    face_recognition: str = "default"
    ocr: str = "tesseract"
    pose: str = "mediapipe"


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    frame_interval: int = 1
    scene_threshold: float = 30.0
    max_video_size_mb: int = 5000
    output_format: str = "mp4"


@dataclass
class MilvusConfig:
    """Configuration for Milvus vector database."""
    host: str = "milvus"
    port: int = 19530
    collection: str = "video_scenes"
    embedding_dim: int = 512
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    nlist: int = 128


@dataclass
class StorageConfig:
    """Configuration for MinIO storage."""
    minio_bucket: str = "doc-ai-vision"
    raw_images_prefix: str = "raw/images/"
    raw_videos_prefix: str = "raw/videos/"
    processed_images_prefix: str = "processed/images/"
    processed_videos_prefix: str = "processed/videos/"
    frames_prefix: str = "frames/"
    scenes_prefix: str = "scenes/"
    metadata_prefix: str = "metadata/"


@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    url: str
    type: str  # 'rest' or 's3'


@dataclass
class CrawlerConfig:
    """Configuration for the automated web crawler."""
    enabled: bool = True
    schedule: str = "0 0 * * *"
    max_items_per_run: int = 1000
    sources: List[SourceConfig] = field(default_factory=list)


@dataclass
class IngestionConfig:
    """Configuration for dataset ingestion."""
    deduplication_enabled: bool = True
    validation_enabled: bool = True
    max_download_workers: int = 4
    chunk_size_mb: int = 100


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "json"
    output: str = "console"


@dataclass
class VisionConfig:
    """Master configuration for the Vision service."""
    grpc_port: int = 50055
    http_port: int = 8002
    models: ModelsConfig = field(default_factory=ModelsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "VisionConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Override with environment variables (VISION_* prefix)
        data = cls._apply_env_overrides(data)

        return cls(
            grpc_port=data.get("vision", {}).get("grpc_port", 50055),
            http_port=data.get("vision", {}).get("http_port", 8002),
            models=ModelsConfig(**data.get("models", {})),
            video=VideoConfig(**data.get("video", {})),
            milvus=MilvusConfig(**data.get("milvus", {})),
            storage=StorageConfig(**data.get("storage", {})),
            crawler=CrawlerConfig(
                **{k: v for k, v in data.get("crawler", {}).items() if k != "sources"},
                sources=[SourceConfig(**src) for src in data.get("crawler", {}).get("sources", [])]
            ),
            ingestion=IngestionConfig(**data.get("ingestion", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables (VISION_*)."""
        # Simple implementation: flatten with double underscores for nesting.
        # Example: VISION_MILVUS__HOST sets milvus.host.
        for key, value in os.environ.items():
            if key.startswith("VISION_"):
                # Remove prefix and split by '__'
                parts = key[6:].lower().split("__")
                target = data
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                # Convert value to appropriate type
                if value.lower() in ("true", "false"):
                    target[parts[-1]] = value.lower() == "true"
                elif value.isdigit():
                    target[parts[-1]] = int(value)
                else:
                    try:
                        target[parts[-1]] = float(value)
                    except ValueError:
                        target[parts[-1]] = value
        return data


# Singleton global config
_config: Optional[VisionConfig] = None


def load_config(config_path: Optional[Path] = None) -> VisionConfig:
    """Load the Vision configuration from a YAML file."""
    if config_path is None:
        config_path = Path("/app/configs/vision.yaml")
    return VisionConfig.from_yaml(config_path)


def get_config(config_path: Optional[Path] = None) -> VisionConfig:
    """Get the global config instance (loads on first call)."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config():
    """Reset the singleton (useful for testing)."""
    global _config
    _config = None