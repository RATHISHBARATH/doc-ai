# ============================================================
# DOC AI DOCA Service – Reflection Reasoning Core
# ============================================================

import logging
from typing import Optional

from src.common.config import DOCAConfig
from src.common.inference_client import InferenceClient
from src.common.models import ReasoningResult

logger = logging.getLogger(__name__)


class Reflection:
    """
    Implements Reflection‑based reasoning.

    The model first generates an initial answer (or reasoning trace),
    then critiques its own output, and finally refines it into a
    more accurate and complete response.
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
        Run Reflection reasoning on the given prompt.

        Args:
            prompt: The input question or task.
            max_tokens: Maximum tokens for the final answer.
            temperature: Sampling temperature.
            include_trace: Whether to return the reasoning steps.

        Returns:
            ReasoningResult with refined answer, confidence, and optional trace.
        """
        logger.info(f"Running Reflection reasoning for prompt: {prompt[:50]}...")

        # 1. Generate an initial answer
        initial_prompt = f"Answer the following question: {prompt}"
        initial_answer = self.inference_client.infer(
            prompt=initial_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not initial_answer:
            logger.error("Reflection: Initial answer generation failed.")
            return ReasoningResult(
                text="Failed to generate initial answer.",
                confidence=0.0,
                reasoning_trace=None,
            )

        # 2. Critique the initial answer
        critique_prompt = (
            f"Critique the following answer for accuracy, completeness, and clarity.\n"
            f"Question: {prompt}\n"
            f"Answer: {initial_answer}\n"
            f"Critique:"
        )

        critique = self.inference_client.infer(
            prompt=critique_prompt,
            max_tokens=150,
            temperature=temperature,
        )

        if not critique:
            logger.warning("Reflection: Critique generation failed; using initial answer.")
            critique = "No critique generated."

        # 3. Refine the answer based on the critique
        refine_prompt = (
            f"Based on the following critique, refine the answer to be more accurate and complete.\n"
            f"Question: {prompt}\n"
            f"Original answer: {initial_answer}\n"
            f"Critique: {critique}\n"
            f"Refined answer:"
        )

        refined_answer = self.inference_client.infer(
            prompt=refine_prompt,
            max_tokens=max_tokens + 50,  # allow for longer refined answer
            temperature=temperature,
        )

        if not refined_answer:
            logger.warning("Reflection: Refinement failed; using initial answer.")
            refined_answer = initial_answer

        # 4. Estimate confidence based on length of critique and refinement
        # A longer critique suggests more thought; a refined answer is usually better.
        confidence = 0.6
        if len(critique) > 50:
            confidence += 0.1
        if len(refined_answer) > len(initial_answer):
            confidence += 0.1
        confidence = min(confidence, 0.95)

        logger.info(f"Reflection completed with confidence={confidence:.2f}")

        return ReasoningResult(
            text=refined_answer,
            confidence=confidence,
            reasoning_trace=[initial_answer, critique, refined_answer] if include_trace else None,
        )