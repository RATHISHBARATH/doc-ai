# ============================================================
# DOC AI Data Pipeline – Production Dockerfile (Final)
# ============================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install system dependencies (gcc/g++ for native extensions)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and configuration
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Stage 2: Runtime
FROM python:3.11-slim

# Create a non‑root user
RUN groupadd -r doc && useradd -r -g doc doc

# Install ca‑certificates for HTTPS downloads
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy source code, configs, and scripts
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/scripts /app/scripts

# Create writable directories for Prefect and Hugging Face cache
RUN mkdir -p /app/.prefect /app/.cache/huggingface && \
    chown -R doc:doc /app /app/src /app/configs /app/scripts /app/.prefect /app/.cache/huggingface && \
    chown -R doc:doc /usr/local/lib/python3.11/site-packages

# Set environment variables for home, caches, and Python path
ENV HOME=/app
ENV PREFECT_HOME=/app/.prefect
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV DATASETS_CACHE=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache
ENV PYTHONPATH=/app

# Switch to the non‑root user
USER doc

# Set the entrypoint to the pipeline runner
ENTRYPOINT ["python", "-m", "src.workflow.pipeline"]