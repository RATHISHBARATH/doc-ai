# ============================================================
# DOC AI DOCA Service – Coordinator Main Entry Point
# ============================================================

import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.config import get_config, DOCAConfig
from src.coordinator.api import router as coordinator_router

logger = logging.getLogger(__name__)


def create_app(config: DOCAConfig) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DOCA Central Intelligence",
        description="Coordinator API for the DOC AI Ecosystem",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the coordinator router
    app.include_router(coordinator_router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "doca"}

    return app


async def start_coordinator():
    """Start the Coordinator service."""
    # Load configuration
    config_path = Path("/app/configs/doca.yaml")
    config = get_config(config_path)

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting DOCA Coordinator...")

    # Create FastAPI app
    app = create_app(config)

    # Run the HTTP server using Uvicorn's Server API
    host = "0.0.0.0"
    port = config.coordinator.http_port

    logger.info(f"Coordinator HTTP server listening on {host}:{port}")

    uvicorn_config = uvicorn.Config(app, host=host, port=port, loop="asyncio")
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(start_coordinator())