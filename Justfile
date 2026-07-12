# Development tasks for doc-ai

# Default: show help
default:
    @just --list

# Build all services
build:
    bazel build //...

# Run all tests
test:
    bazel test //...

# Start the system with Docker Compose
up:
    docker compose up -d

# Stop the system
down:
    docker compose down

# View logs
logs:
    docker compose logs -f

# Clean Bazel cache
clean:
    bazel clean