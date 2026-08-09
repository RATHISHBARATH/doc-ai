# ============================================================
# DOC AI Gateway – Production Dockerfile (Corrected)
# ============================================================

# Stage 1: Builder
FROM rust:slim AS builder

# Install system dependencies (protoc, OpenSSL, pkg-config)
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project (monorepo)
COPY . .

# Build the gateway in release mode
WORKDIR /app/services/gateway
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

# Create a non‑root user for security
RUN groupadd -r doc && useradd -r -g doc doc

# Install runtime dependencies: ca‑certificates and curl (for healthcheck)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder
COPY --from=builder /app/services/gateway/target/release/gateway /usr/local/bin/gateway

# Change ownership and switch to non‑root user
RUN chown doc:doc /usr/local/bin/gateway
USER doc

# Expose the gateway port
EXPOSE 8080

# Health check (uses curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["gateway"]