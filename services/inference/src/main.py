# ============================================================
# Inference Service – Unified Entry Point (FastAPI + gRPC)
# ============================================================

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import config
from src.model_loader import Model
from src.grpc_server import serve as grpc_serve

logger = logging.getLogger(__name__)

# ============================================================
# Telemetry setup (shared between FastAPI and gRPC)
# ============================================================

def init_telemetry():
    provider = TracerProvider()
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=config.otel_endpoint)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

# ============================================================
# FastAPI app
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model and start gRPC server
    logger.info("Starting inference service...")
    init_telemetry()
    model = Model(
        model_name=config.model_name,
        use_gpu=config.use_gpu,
        quantize=config.quantize,
    )
    app.state.model = model

    # Start gRPC server as a background task
    grpc_task = asyncio.create_task(grpc_serve(model))
    app.state.grpc_task = grpc_task
    logger.info(f"gRPC server starting on port {config.grpc_port}")

    yield
    # Shutdown: clean up gRPC task
    grpc_task.cancel()
    await grpc_task
    logger.info("Inference service shut down")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return {"metrics": "not implemented yet"}

FastAPIInstrumentor.instrument_app(app)

# ============================================================
# Run the service
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.http_port,
        log_level="info",
    )