# ============================================================
# DOC AI Auth – Production Dockerfile (Corrected)
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

# Build the auth service in release mode
WORKDIR /app/services/auth
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

# Create non‑root user
RUN groupadd -r doc && useradd -r -g doc doc

# Install runtime dependencies: ca‑certificates (curl optional)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install grpc-health-probe for gRPC health checks
ADD https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.28/grpc_health_probe-linux-amd64 /usr/local/bin/grpc-health-probe
RUN chmod +x /usr/local/bin/grpc-health-probe

# Copy the binary
COPY --from=builder /app/services/auth/target/release/auth /usr/local/bin/auth

# Change ownership and switch to non‑root user
RUN chown doc:doc /usr/local/bin/auth /usr/local/bin/grpc-health-probe
USER doc

# Expose ports
EXPOSE 50051 50052

# Health check – use gRPC health probe on port 50051
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD grpc-health-probe -addr=localhost:50051 || exit 1

# Set entrypoint
ENTRYPOINT ["auth"]