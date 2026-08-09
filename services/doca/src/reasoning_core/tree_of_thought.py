# ============================================================
# DOC AI DOCA Service – Tree‑of‑Thought Reasoning Core
# ============================================================

import logging
import asyncio
from typing import List, Optional, Dict, Any

from src.common.config import DOCAConfig
from src.common.inference_client import InferenceClient
from src.common.models import ReasoningResult

logger = logging.getLogger(__name__)


class TreeOfThought:
    """
    Implements Tree‑of‑Thought (ToT) reasoning.

    ToT explores multiple reasoning branches, evaluates each branch,
    and selects the best one based on a scoring heuristic.
    """

    def __init__(self, inference_client: InferenceClient, config: DOCAConfig):
        self.inference_client = inference_client
        self.config = config

    async def run(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        num_branches: int = 3,
        depth: int = 1,
    ) -> ReasoningResult:
        """
        Run Tree‑of‑Thought reasoning on the given prompt.

        Args:
            prompt: The input question or task.
            max_tokens: Maximum tokens for the final answer.
            temperature: Sampling temperature.
            num_branches: Number of parallel reasoning branches.
            depth: Number of expansion steps per branch.

        Returns:
            ReasoningResult with the best answer, confidence, and trace.
        """
        logger.info(f"Running ToT reasoning with {num_branches} branches, depth {depth}")

        # 1. Generate initial reasoning steps (first level)
        initial_prompt = (
            f"Let's think step by step to answer the following question.\n"
            f"Question: {prompt}\n"
            f"Reasoning:"
        )

        # Generate multiple initial steps by sampling with higher temperature
        initial_steps = []
        for i in range(num_branches):
            step = self.inference_client.infer(
                prompt=initial_prompt,
                max_tokens=100,
                temperature=temperature + 0.1,  # slightly higher for diversity
            )
            if step:
                initial_steps.append(step)
            else:
                logger.warning(f"Branch {i} failed to produce initial step.")

        if not initial_steps:
            logger.error("ToT: No initial reasoning steps generated.")
            return ReasoningResult(
                text="Unable to generate reasoning branches.",
                confidence=0.0,
                reasoning_trace=None,
            )

        # 2. Expand each branch to depth
        branches = []
        for idx, step in enumerate(initial_steps):
            branch_trace = [step]
            current_prompt = step

            for d in range(depth - 1):
                expand_prompt = (
                    f"Continue the reasoning:\n{current_prompt}\n"
                    f"Next step:"
                )
                next_step = self.inference_client.infer(
                    prompt=expand_prompt,
                    max_tokens=100,
                    temperature=temperature,
                )
                if next_step:
                    branch_trace.append(next_step)
                    current_prompt = next_step
                else:
                    logger.warning(f"Branch {idx} stopped expanding at depth {d+1}.")
                    break

            # Combine trace into a single reasoning string
            full_trace = "\n".join(branch_trace)
            branches.append({
                "trace": branch_trace,
                "full_trace": full_trace,
                "score": 0.0,
            })

        # 3. Score each branch using a self‑evaluation prompt
        for branch in branches:
            eval_prompt = (
                f"Evaluate the following reasoning for correctness and completeness:\n"
                f"{branch['full_trace']}\n"
                f"Score (0-10):"
            )
            score_text = self.inference_client.infer(
                prompt=eval_prompt,
                max_tokens=10,
                temperature=0.2,
            )
            try:
                score = float(score_text.strip())
                branch["score"] = max(0.0, min(10.0, score)) / 10.0  # normalize to 0-1
            except:
                logger.warning(f"Failed to parse score from: {score_text}")
                branch["score"] = 0.5  # fallback

        # 4. Select the best branch
        best_branch = max(branches, key=lambda b: b["score"])
        best_trace = best_branch["trace"]
        best_full_trace = best_branch["full_trace"]
        best_score = best_branch["score"]

        logger.info(f"ToT selected branch with score {best_score:.2f}")

        # 5. Generate final answer based on the best trace
        answer_prompt = (
            f"Based on the following reasoning:\n{best_full_trace}\n\n"
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

        # 6. Confidence is based on branch score
        confidence = best_score

        return ReasoningResult(
            text=final_answer,
            confidence=confidence,
            reasoning_trace=best_trace,
            alternatives=[b["full_trace"] for b in branches if b != best_branch],
        )