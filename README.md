# DOC AI Ecosystem

A production-grade AI Operating System with custom LLMs, multi-agent orchestration, and robotics integration.

## Getting Started

1. Run `make setup` to install dependencies.
2. Run `make build` to build Docker images.
3. Run `docker compose -f infrastructure/compose/docker-compose.yml up -d` to start the stack.
4. Visit `http://localhost:8080/health` to verify the gateway.

See `docs/` for detailed documentation.
