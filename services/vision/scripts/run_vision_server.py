# ============================================================
# DOC AI Vision Service – Entry Point Script
# ============================================================

import asyncio
import logging
import threading
import signal
import sys

from src.api.grpc_server import serve_grpc, run_sync_grpc
from src.api.http_server import run_http
from src.common.config import get_config

logger = logging.getLogger(__name__)


def run_servers() -> None:
    """
    Run both the gRPC and HTTP servers concurrently.
    The gRPC server runs in a separate thread, the HTTP server runs in the main thread.
    """
    config = get_config()
    grpc_port = config.grpc_port
    http_port = config.http_port

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting Vision service: gRPC port={grpc_port}, HTTP port={http_port}")

    # Run gRPC server in a background thread (since it's blocking)
    grpc_thread = threading.Thread(
        target=lambda: run_sync_grpc(None, port=grpc_port),
        daemon=True,
    )
    grpc_thread.start()
    logger.info(f"gRPC server started in background thread on port {grpc_port}")

    # Run HTTP server in the main thread (uvicorn is blocking)
    logger.info(f"Starting HTTP server on port {http_port}")
    run_http(host="0.0.0.0", port=http_port)


def main() -> None:
    """
    Main entry point: set up signal handlers and run servers.
    """
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_servers()


if __name__ == "__main__":
    main()