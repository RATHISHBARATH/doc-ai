# ============================================================
# DOC AI Inference – Production Dockerfile (with Configs)
# ============================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install system dependencies (gcc/g++ for native extensions, protoc for protobuf)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file (without torch – torch will be installed separately)
COPY services/inference/requirements.txt .

# Install torch (CPU‑only – smaller, faster build; can be changed to CUDA later)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY services/inference/src/ ./src/

# Copy the protobuf definition
COPY shared/proto/inference.proto ./shared/proto/inference.proto

# Generate gRPC stubs directly into /app
RUN python -m grpc_tools.protoc \
    -I shared/proto \
    --python_out=. \
    --grpc_python_out=. \
    shared/proto/inference.proto

# Stage 2: Runtime
FROM python:3.11-slim

# Install ca-certificates and curl (for healthcheck)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non‑root user with home directory
RUN groupadd -r doc && useradd -r -g doc -m doc

WORKDIR /app

# Ensure /app/src and /app/configs exist
RUN mkdir -p /app/src /app/cache /app/configs

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy source code
COPY --from=builder /app/src /app/src

# Copy generated protobuf stubs
COPY --from=builder /app/inference_pb2*.py /app/

# Copy configuration files
COPY services/inference/configs/ /app/configs/

# Change ownership of everything to doc
RUN chown -R doc:doc /usr/local/lib/python3.11/site-packages /app /home/doc

# Switch to non‑root user
USER doc

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV OTEL_EXPORTER_OTLP_ENDPOINT=
ENV DOC_CONFIG_PATH=/app/configs/development.yaml

# Expose ports
EXPOSE 8000 50053

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Run the unified entry point
CMD ["python", "-m", "src.main"]