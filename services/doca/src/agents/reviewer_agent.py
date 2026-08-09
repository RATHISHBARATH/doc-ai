# ============================================================
# DOC AI DOCA Service – Reviewer Agent
# ============================================================

import logging
from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.common.models import AgentType, Task, ReasoningResult
from src.common.config import DOCAConfig
from src.common.inference_client import InferenceClient

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """
    Specialized agent that reviews and critiques answers produced by other agents.
    It checks for logical consistency, factual accuracy, and completeness,
    and provides a critique score and suggestions for improvement.
    """

    def __init__(
        self,
        agent_id: str = None,
        name: str = "ReviewerAgent",
        description: str = "Reviews and critiques answers for quality",
        config: Optional[DOCAConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.REVIEWER,
            name=name,
            description=description,
            config=config,
        )
        self.inference_client = InferenceClient(self.config)

    async def _execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute a review task.

        The task input should contain:
        - 'answer': the answer to be reviewed (string)
        - 'prompt': the original question (string)
        - 'reasoning_trace': optional list of reasoning steps

        Returns a dict with:
        - 'score': a float between 0 and 1
        - 'critique': a string containing the critique
        - 'suggestions': a list of suggested improvements
        """
        input_data = task.input_data
        answer = input_data.get("answer")
        prompt = input_data.get("prompt")
        reasoning_trace = input_data.get("reasoning_trace", [])

        if not answer:
            raise ValueError("Task input missing 'answer'")
        if not prompt:
            raise ValueError("Task input missing 'prompt'")

        self.logger.info(f"Reviewing answer for prompt: {prompt[:50]}...")

        # Build a review prompt
        trace_text = ""
        if reasoning_trace:
            trace_text = "\n".join(reasoning_trace)
            trace_text = f"\nReasoning trace:\n{trace_text}"

        review_prompt = (
            f"Review the following answer for logical consistency, factual accuracy, and completeness.\n"
            f"Question: {prompt}\n"
            f"Answer: {answer}\n"
            f"{trace_text}\n"
            f"Provide a critique, a score from 0 to 10, and specific suggestions for improvement.\n"
            f"Format: Score: X/10\nCritique: ...\nSuggestions: ..."
        )

        review_output = self.inference_client.generate(
            prompt=review_prompt,
            max_tokens=200,
            temperature=0.3,  # lower temperature for more deterministic critique
        )

        if not review_output:
            self.logger.warning("Review generation failed.")
            return {
                "score": 0.5,
                "critique": "Unable to generate critique.",
                "suggestions": [],
            }

        # Parse the output (simple parsing – can be improved with regex)
        score = 0.5
        critique = review_output
        suggestions = []

        try:
            lines = review_output.split("\n")
            for line in lines:
                if line.lower().startswith("score:"):
                    score_str = line.split(":")[1].strip().split("/")[0].strip()
                    score = float(score_str) / 10.0
                    score = max(0.0, min(1.0, score))
                elif line.lower().startswith("critique:"):
                    critique = line.split(":", 1)[1].strip()
                elif line.lower().startswith("suggestions:"):
                    sugg = line.split(":", 1)[1].strip()
                    suggestions = [s.strip() for s in sugg.split(",") if s.strip()]
        except Exception as e:
            self.logger.warning(f"Failed to parse review output: {e}")

        # If parsing failed, keep the defaults but log the raw output
        if score == 0.5 and not suggestions and critique == review_output:
            self.logger.debug(f"Raw review output: {review_output}")

        self.logger.info(f"Review completed with score: {score:.2f}")

        return {
            "score": score,
            "critique": critique,
            "suggestions": suggestions,
        }