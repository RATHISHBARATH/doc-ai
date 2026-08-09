# ============================================================
# DOC AI DOCA Service – Production Dockerfile (Project Root Context)
# ============================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY services/doca/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade protobuf

# Copy DOCA source, configs, scripts
COPY services/doca/src/ ./src/
COPY services/doca/configs/ ./configs/
COPY services/doca/scripts/ ./scripts/

# Copy shared Python package
COPY shared/python/src/doc_ai_common/ ./doc_ai_common/

# Stage 2: Runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r doc && useradd -r -g doc -m doc
WORKDIR /app
RUN mkdir -p /app/src /app/configs /app/data

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/doc_ai_common /app/doc_ai_common

RUN chown -R doc:doc /usr/local/lib/python3.11/site-packages /app /home/doc
USER doc

ENV PYTHONPATH=/app
ENV DOCA_CONFIG_PATH=/app/configs/doca.yaml
ENV DOCA_NATS_URL=nats://nats:4222
ENV DOCA_REDIS_URL=redis://redis:6379
ENV DOCA_POSTGRES_URL=postgres://doc_user:${POSTGRES_PASSWORD:-changeme}@postgres:5432/doc_ai
ENV DOCA_MINIO_ENDPOINT=minio:9000
ENV DOCA_MINIO_ACCESS_KEY=${MINIO_ROOT_USER:-minioadmin}
ENV DOCA_MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD:-changeme}
ENV DOCA_MINIO_BUCKET=doc-ai-data
ENV DOCA_INFERENCE_ADDR=inference:50053
ENV DOCA_LOG_LEVEL=INFO

EXPOSE 8001 50054
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1
CMD ["python", "-m", "src.coordinator.main"]