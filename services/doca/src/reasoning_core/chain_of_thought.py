# ============================================================
# DOC AI DOCA Service – Chain‑of‑Thought Reasoning Core
# ============================================================

import logging
from typing import List, Optional

from src.common.config import DOCAConfig
from src.common.inference_client import InferenceClient
from src.common.models import ReasoningResult

logger = logging.getLogger(__name__)


class ChainOfThought:
    """
    Implements Chain‑of‑Thought (CoT) reasoning.

    The model is prompted to generate a step‑by‑step reasoning trace before
    producing the final answer. This improves logical consistency and
    interpretability.
    """

    def __init__(self, inference_client: InferenceClient, config: DOCAConfig):
        self.inference_client = inference_client
        self.config = config

    async def run(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        include_trace: bool = True,
    ) -> ReasoningResult:
        """
        Run Chain‑of‑Thought reasoning on the given prompt.

        Args:
            prompt: The input question or task.
            max_tokens: Maximum tokens for the final answer.
            temperature: Sampling temperature.
            include_trace: Whether to return the reasoning steps.

        Returns:
            ReasoningResult with answer, confidence, and optional trace.
        """
        logger.info(f"Running CoT reasoning for prompt: {prompt[:50]}...")

        # 1. Generate reasoning trace
        trace_prompt = (
            f"Let's think step by step to answer the following question.\n"
            f"Question: {prompt}\n"
            f"Reasoning:"
        )

        trace_text = self.inference_client.infer(
            prompt=trace_prompt,
            max_tokens=200,   # longer for reasoning
            temperature=temperature,
        )

        if not trace_text:
            logger.error("CoT reasoning step produced no output.")
            return ReasoningResult(
                text="I couldn't reason through that.",
                confidence=0.0,
                reasoning_trace=None,
            )

        # 2. Parse trace into steps (rough splitting by newlines or periods)
        steps = [s.strip() for s in trace_text.split("\n") if s.strip()]
        if not steps:
            steps = [trace_text.strip()]

        # 3. Generate final answer based on the trace
        answer_prompt = (
            f"Based on the following reasoning:\n{trace_text}\n\n"
            f"Now, provide a concise final answer to: {prompt}\n"
            f"Final answer:"
        )

        final_answer = self.inference_client.infer(
            prompt=answer_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not final_answer:
            final_answer = "Unable to generate final answer."

        # 4. Estimate confidence (simple heuristic: length of reasoning trace)
        confidence = min(0.9, 0.5 + (len(steps) * 0.05))

        logger.info(f"CoT generated {len(steps)} reasoning steps, confidence={confidence:.2f}")

        return ReasoningResult(
            text=final_answer,
            confidence=confidence,
            reasoning_trace=steps if include_trace else None,
        )