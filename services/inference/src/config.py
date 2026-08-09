# ============================================================
# Inference Service Configuration Module
# ============================================================

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # gRPC server port
    grpc_port: int = 50053

    # HTTP port for health and metrics (FastAPI)
    http_port: int = 8000

    # Model to load (Hugging Face model ID or local path)
    model_name: str = "distilgpt2"

    # Whether to use GPU (auto, true, false)
    use_gpu: str = "auto"

    # Quantization level (none, 4bit, 8bit)
    quantize: str = "none"

    # Max batch size for dynamic batching
    max_batch_size: int = 8

    # Batch timeout in milliseconds (how long to wait for more requests)
    batch_timeout_ms: int = 50

    # Maximum tokens to generate (default)
    max_tokens_default: int = 128

    # Default temperature
    temperature_default: float = 0.8

    # OpenTelemetry service name
    otel_service_name: str = "inference"

    # OpenTelemetry OTLP endpoint (Jaeger)
    otel_endpoint: str = "http://jaeger:4317"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            grpc_port=int(os.environ.get("INFERENCE_GRPC_PORT", "50053")),
            http_port=int(os.environ.get("INFERENCE_HTTP_PORT", "8000")),
            model_name=os.environ.get("MODEL_NAME", "distilgpt2"),
            use_gpu=os.environ.get("USE_GPU", "auto"),
            quantize=os.environ.get("MODEL_QUANTIZE", "none"),
            max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "8")),
            batch_timeout_ms=int(os.environ.get("BATCH_TIMEOUT_MS", "50")),
            max_tokens_default=int(os.environ.get("MAX_TOKENS_DEFAULT", "128")),
            temperature_default=float(os.environ.get("TEMPERATURE_DEFAULT", "0.8")),
            otel_service_name=os.environ.get("OTEL_SERVICE_NAME", "inference"),
            otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"),
        )


# Global singleton config instance
config = Config.from_env()