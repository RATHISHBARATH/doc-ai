# ============================================================
# DOC AI Vision Service – HTTP (FastAPI) Server
# ============================================================

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from starlette.middleware.cors import CORSMiddleware

from src.common.config import get_config, VisionConfig
from src.common.models import VisionTaskType
from src.core.orchestrator import VisionOrchestrator
from src.crawler.scheduler import CrawlerScheduler

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("vision_http_requests_total", "Total HTTP requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("vision_http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for startup and shutdown hooks.
    """
    # Startup: load configuration, initialize orchestrator, start crawler
    config = get_config()
    app.state.config = config
    app.state.orchestrator = VisionOrchestrator()

    # Start crawler scheduler if enabled
    app.state.crawler = CrawlerScheduler(config)
    if config.crawler.enabled:
        app.state.crawler.start()
        logger.info("Crawler scheduler started on application startup")

    logger.info("Vision HTTP server started")

    yield

    # Shutdown: stop crawler and clean up
    if hasattr(app.state, "crawler"):
        await app.state.crawler.stop()
        logger.info("Crawler scheduler stopped on shutdown")

    # Close any other resources (e.g., aiohttp sessions)
    if hasattr(app.state, "orchestrator"):
        # No explicit close method yet, but we can add one later
        pass

    logger.info("Vision HTTP server shut down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="DOC AI Vision Service",
        version="0.1.0",
        description="Computer vision and multimodal perception microservice for the DOC AI Ecosystem.",
        lifespan=lifespan,
    )

    # Add CORS middleware (allow all for development; restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health check endpoint
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        """Health check endpoint for Kubernetes / Docker."""
        return {"status": "healthy", "service": "vision"}

    # ------------------------------------------------------------------
    # Metrics endpoint (Prometheus)
    # ------------------------------------------------------------------

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return generate_latest(REGISTRY)

    # ------------------------------------------------------------------
    # API endpoints (simple REST wrappers around gRPC methods)
    # ------------------------------------------------------------------

    @app.post("/api/v1/detect")
    async def detect_objects(image_data: bytes):
        """
        Detect objects in an uploaded image.
        Expects raw image bytes in the request body.
        """
        with REQUEST_LATENCY.labels(method="POST", endpoint="/detect").time():
            REQUEST_COUNT.labels(method="POST", endpoint="/detect").inc()
            orchestrator: VisionOrchestrator = app.state.orchestrator
            try:
                result = await orchestrator.process_image(
                    image_data=image_data,
                    tasks=[VisionTaskType.OBJECT_DETECTION],
                    store_result=True,
                )
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Detect objects error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/ocr")
    async def ocr(image_data: bytes):
        """
        Perform OCR on an uploaded image.
        Expects raw image bytes in the request body.
        """
        with REQUEST_LATENCY.labels(method="POST", endpoint="/ocr").time():
            REQUEST_COUNT.labels(method="POST", endpoint="/ocr").inc()
            orchestrator: VisionOrchestrator = app.state.orchestrator
            try:
                result = await orchestrator.process_image(
                    image_data=image_data,
                    tasks=[VisionTaskType.OCR],
                    store_result=True,
                )
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"OCR error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/faces")
    async def recognize_faces(image_data: bytes):
        """
        Detect and recognize faces in an uploaded image.
        Expects raw image bytes in the request body.
        """
        with REQUEST_LATENCY.labels(method="POST", endpoint="/faces").time():
            REQUEST_COUNT.labels(method="POST", endpoint="/faces").inc()
            orchestrator: VisionOrchestrator = app.state.orchestrator
            try:
                result = await orchestrator.process_image(
                    image_data=image_data,
                    tasks=[VisionTaskType.FACE_RECOGNITION],
                    store_result=True,
                )
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Face recognition error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/process_video")
    async def process_video(video_url: str, extract_frames: bool = True, segment_scenes: bool = True):
        """
        Process a video: extract frames, segment scenes, and run vision tasks.
        Expects a JSON body with:
        {
            "video_url": "path or URL to video",
            "extract_frames": true,
            "segment_scenes": true
        }
        """
        with REQUEST_LATENCY.labels(method="POST", endpoint="/process_video").time():
            REQUEST_COUNT.labels(method="POST", endpoint="/process_video").inc()
            orchestrator: VisionOrchestrator = app.state.orchestrator
            try:
                result = await orchestrator.process_video(
                    video_url=video_url,
                    extract_frames=extract_frames,
                    segment_scenes=segment_scenes,
                    store_result=True,
                )
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Process video error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Crawler control endpoints (optional, for debugging)
    # ------------------------------------------------------------------

    @app.post("/api/v1/crawler/start")
    async def start_crawler():
        """Start the crawler scheduler (if not already running)."""
        crawler: CrawlerScheduler = app.state.crawler
        if crawler.is_running():
            return {"status": "already_running"}
        crawler.start()
        return {"status": "started"}

    @app.post("/api/v1/crawler/stop")
    async def stop_crawler():
        """Stop the crawler scheduler."""
        crawler: CrawlerScheduler = app.state.crawler
        if not crawler.is_running():
            return {"status": "not_running"}
        await crawler.stop()
        return {"status": "stopped"}

    @app.get("/api/v1/crawler/status")
    async def crawler_status():
        """Get the current status of the crawler scheduler."""
        crawler: CrawlerScheduler = app.state.crawler
        return {"running": crawler.is_running()}

    return app


def run_http(host: str = "0.0.0.0", port: int = 8002) -> None:
    """
    Run the HTTP server using uvicorn.
    """
    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


# For direct execution (e.g., python -m src.api.http_server)
if __name__ == "__main__":
    run_http()