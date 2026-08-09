# ============================================================
# Inference Service – gRPC Server (Final)
# ============================================================

import asyncio
import logging
import sys
import time
from typing import AsyncGenerator

import grpc
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import config
from src.model_loader import Model
from src.batcher import Batcher, BatchRequest
from src.streamer import stream_generate

# Add the proto_gen directory to Python's path so we can import the generated modules
sys.path.append("/app/proto_gen")

# Import generated protobuf code (now top‑level modules)
import inference_pb2
import inference_pb2_grpc

logger = logging.getLogger(__name__)

# ============================================================
# gRPC Servicer Implementation
# ============================================================

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    """Implements the Inference gRPC service with batching and streaming."""

    def __init__(self, model: Model):
        self.model = model
        self.tracer = trace.get_tracer(__name__)

        # Initialize the batcher with the batch processing function
        self.batcher = Batcher(
            process_batch_fn=self._process_batch,
            max_batch_size=config.max_batch_size,
            batch_timeout_ms=config.batch_timeout_ms,
        )
        self.batcher.start()

    async def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Process a batch of requests using the model."""
        # Extract prompts and parameters
        prompts = [req.prompt for req in batch]
        max_tokens = batch[0].max_tokens  # Assume all requests have the same max_tokens
        temperature = batch[0].temperature

        # Run the model on the batch (blocking, offloaded to thread)
        results = await asyncio.to_thread(
            self.model.generate_batch,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Set the results on each request's future
        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)

    async def Infer(
        self,
        request: inference_pb2.InferRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.InferResponse:
        """Unary inference with batching."""
        with self.tracer.start_as_current_span("Infer"):
            start_time = time.time()
            prompt = request.prompt
            max_tokens = request.max_tokens or config.max_tokens_default
            temperature = request.temperature or config.temperature_default

            logger.info(f"Infer request: prompt='{prompt[:50]}...', max_tokens={max_tokens}")

            # Submit to batcher and wait for result
            text = await self.batcher.submit(prompt, max_tokens, temperature)

            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Infer completed in {latency_ms}ms, {len(text.split())} tokens")

            return inference_pb2.InferResponse(
                text=text,
                confidence=0.8,
                latency_ms=latency_ms,
            )

    async def StreamInfer(
        self,
        request: inference_pb2.InferRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[inference_pb2.InferResponse, None]:
        """Streaming inference without batching (for low latency)."""
        with self.tracer.start_as_current_span("StreamInfer"):
            prompt = request.prompt
            max_tokens = request.max_tokens or config.max_tokens_default
            temperature = request.temperature or config.temperature_default

            logger.info(f"StreamInfer request: prompt='{prompt[:50]}...', max_tokens={max_tokens}")

            # Generate tokens with streaming
            async for token in stream_generate(self.model, prompt, max_tokens, temperature):
                yield inference_pb2.InferResponse(
                    text=token,
                    confidence=0.8,
                    latency_ms=0,
                )

            logger.info("StreamInfer completed")


# ============================================================
# Telemetry Initialization
# ============================================================

def init_telemetry() -> None:
    """Initialize OpenTelemetry tracing."""
    provider = TracerProvider()
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=config.otel_endpoint)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Instrument gRPC server
    GrpcInstrumentorServer().instrument()


# ============================================================
# Server Runner
# ============================================================

async def serve(model: Model) -> None:
    """Start the gRPC server with the given model."""
    init_telemetry()
    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(model),
        server,
    )
    listen_addr = f"[::]:{config.grpc_port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"gRPC server starting on {listen_addr}")

    await server.start()
    logger.info("gRPC server started successfully")
    await server.wait_for_termination()