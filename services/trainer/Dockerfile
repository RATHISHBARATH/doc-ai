# ============================================================
# DOC AI Trainer – Production Dockerfile
# ============================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install system dependencies (gcc/g++ for native extensions)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies (ca-certificates for HTTPS, curl for healthchecks)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non‑root user
RUN groupadd -r doc && useradd -r -g doc -m doc

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy source, configs, and scripts
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/scripts ./scripts

# Ensure /app/cache exists for model checkpoints and HuggingFace cache
RUN mkdir -p /app/cache

# Change ownership of everything to doc
RUN chown -R doc:doc /usr/local/lib/python3.11/site-packages /app

# Switch to non‑root user
USER doc

# Set environment variables for cache and logging
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV PYTHONPATH=/app

# Default command: run the trainer script
CMD ["python", "-m", "src.trainer"]