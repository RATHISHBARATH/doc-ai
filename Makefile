.PHONY: help setup build test clean proto

help:
	@echo "Available targets:"
	@echo "  setup    - Install all dependencies"
	@echo "  build    - Build all services"
	@echo "  test     - Run all tests"
	@echo "  clean    - Clean build artifacts"
	@echo "  proto    - Generate gRPC code"

setup:
	@echo "Running setup scripts..."
	bash scripts/setup.sh

build:
	docker compose -f infrastructure/compose/docker-compose.yml build

test:
	cd services/gateway && cargo test
	cd services/auth && cargo test
	cd services/inference && pytest

clean:
	cargo clean
	rm -rf services/**/target
	find . -type d -name "__pycache__" -exec rm -rf {} +

proto:
	bash scripts/gen-proto.sh
