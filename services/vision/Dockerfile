# ============================================================
# DOC AI Vision Service – Production Dockerfile (Final)
# ============================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set CMake policy to avoid version mismatch error
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN apt-get update && apt-get install -y \
    gcc g++ build-essential cmake protobuf-compiler \
    libgl1 libsm6 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY services/vision/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared doc_ai_common package (clear any stale stubs)
COPY shared/python/src/doc_ai_common/ ./doc_ai_common/
RUN rm -f ./doc_ai_common/*_pb2*.py

# Copy source code, configs, scripts
COPY services/vision/src/ ./src/
COPY services/vision/configs/ ./configs/
COPY services/vision/scripts/ ./scripts/

# Copy proto files and generate ONLY vision.proto stubs into doc_ai_common
COPY shared/proto/ ./proto/
RUN python -m grpc_tools.protoc -I ./proto \
    --python_out=./doc_ai_common \
    --grpc_python_out=./doc_ai_common \
    ./proto/vision.proto

# Fix the import in vision_pb2_grpc.py to use relative import
RUN sed -i 's/import vision_pb2 as vision__pb2/from . import vision_pb2 as vision__pb2/' ./doc_ai_common/vision_pb2_grpc.py

# Overwrite doc_ai_common/__init__.py to only export vision modules
RUN cat > ./doc_ai_common/__init__.py << 'EOF'
# ============================================================
# DOC AI Common – Vision-only stubs
# ============================================================
from .vision_pb2 import *
from .vision_pb2_grpc import *
__all__ = [
"DetectRequest", "DetectResponse", "Object",
"OCRRequest", "OCRResponse",
"FaceRequest", "FaceResponse", "Face",
"VideoRequest", "VideoResponse", "Scene",
"QueryRequest", "QueryResponse",
"VisionStub", "VisionServicer",
]
EOF

# Stage 2: Runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ca-certificates curl ffmpeg libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r doc && useradd -r -g doc -m doc

WORKDIR /app
RUN mkdir -p /app/src /app/configs /app/data

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/doc_ai_common ./doc_ai_common

RUN chown -R doc:doc /usr/local/lib/python3.11/site-packages /app /home/doc
USER doc

ENV PYTHONPATH=/app
ENV VISION_CONFIG_PATH=/app/configs/vision.yaml
ENV VISION_NATS_URL=nats://nats:4222
ENV VISION_REDIS_URL=redis://redis:6379
ENV VISION_MINIO_ENDPOINT=minio:9000
ENV VISION_MINIO_ACCESS_KEY=${MINIO_ROOT_USER:-minioadmin}
ENV VISION_MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD:-changeme}
ENV VISION_MINIO_BUCKET=doc-ai-vision
ENV VISION_MILVUS_HOST=milvus
ENV VISION_MILVUS_PORT=19530
ENV VISION_LOG_LEVEL=INFO
ENV VISION_OTEL_ENDPOINT=http://jaeger:4317

EXPOSE 50055 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-m", "src.api.grpc_server"]