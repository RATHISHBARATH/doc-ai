# ============================================================
# Inference Service – Dynamic Batching Engine
# ============================================================

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Callable, Awaitable, Optional

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single inference request waiting in the batch queue."""
    prompt: str
    max_tokens: int
    temperature: float
    future: asyncio.Future  # Will be set with the result


class Batcher:
    """
    Dynamic batcher that collects requests and processes them in batches.
    
    Usage:
        batcher = Batcher(process_batch_fn)
        batcher.start()
        result = await batcher.submit(prompt, max_tokens, temperature)
    """

    def __init__(
        self,
        process_batch_fn: Callable[[List[BatchRequest]], Awaitable[None]],
        max_batch_size: int = 8,
        batch_timeout_ms: int = 50,
    ):
        self.process_batch_fn = process_batch_fn
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        self.queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._is_running = False

    def start(self) -> None:
        """Start the background batch processor task."""
        if self._is_running:
            return
        self._is_running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._batch_processor_loop())
        logger.info("Batcher started")

    async def stop(self) -> None:
        """Gracefully stop the batcher and process any remaining requests."""
        if not self._is_running:
            return
        self._stop_event.set()
        if self._task:
            await self._task
        self._is_running = False
        logger.info("Batcher stopped")

    async def submit(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Submit a request to the batch queue and wait for the result.
        Returns the generated text.
        """
        future = asyncio.get_running_loop().create_future()
        req = BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            future=future,
        )
        await self.queue.put(req)
        return await future

    async def _batch_processor_loop(self) -> None:
        """Background loop that collects requests and processes them in batches."""
        while not self._stop_event.is_set():
            try:
                # Wait for the first request to arrive
                first_req = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.1,  # 100ms check interval
                )
            except asyncio.TimeoutError:
                continue  # No requests, loop again

            # Collect all requests that arrive within the batch window
            batch: List[BatchRequest] = [first_req]
            start_time = time.time()
            timeout = self.batch_timeout_ms / 1000.0

            while len(batch) < self.max_batch_size:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    break

                try:
                    req = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining,
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break  # Window closed

            # Process the batch (logging optional)
            logger.debug(f"Processing batch of {len(batch)} requests")
            try:
                await self.process_batch_fn(batch)
            except Exception as e:
                logger.exception(f"Batch processing failed: {e}")
                # Mark all futures with the exception
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)