# ============================================================
# Inference Service – Token Streaming
# ============================================================

import asyncio
import logging
import torch  # <-- added missing import
from queue import Queue
from typing import AsyncGenerator, Optional

from transformers import TextIteratorStreamer

from src.model_loader import Model
from src.config import config

logger = logging.getLogger(__name__)


async def stream_generate(
    model: Model,
    prompt: str,
    max_tokens: int,
    temperature: float,
    queue_size: int = 10,
) -> AsyncGenerator[str, None]:
    """
    Generate tokens from the model and yield them one by one asynchronously.
    
    Uses Hugging Face's TextIteratorStreamer to receive tokens in real time.
    The blocking generation runs in a separate thread to keep the async loop free.
    
    Args:
        model: The loaded model wrapper.
        prompt: Input prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        queue_size: Size of the queue between the generator thread and the async loop.
    
    Yields:
        Tokens as they are generated.
    """
    # Prepare the streamer
    streamer = TextIteratorStreamer(
        model.tokenizer,
        skip_prompt=True,
        timeout=None,
    )

    # Prepare the inputs
    inputs = model.tokenizer(prompt, return_tensors="pt")
    if model.device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Define the blocking generation function
    def generate():
        with torch.no_grad():
            model.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=model.tokenizer.eos_token_id,
                streamer=streamer,
            )

    # Run the generation in a separate thread
    loop = asyncio.get_running_loop()
    generate_task = loop.run_in_executor(None, generate)

    # Yield tokens from the streamer asynchronously
    try:
        for token in streamer:
            # Yield each token as it arrives
            yield token
    finally:
        # Ensure the generation thread is cleaned up
        await asyncio.sleep(0)  # yield control to the event loop
        # The streamer's queue is exhausted; the generation thread will finish.
        # Wait for the thread to actually complete (optional)
        await generate_task