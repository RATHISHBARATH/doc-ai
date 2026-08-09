# ============================================================
# DOC AI DOCA Service – Reasoning Agent
# ============================================================

import logging
from typing import Any, Dict, Optional

from src.agents.base_agent import BaseAgent
from src.common.models import AgentType, Task, ReasoningResult
from src.common.config import DOCAConfig
from src.reasoning_core.chain_of_thought import ChainOfThought
from src.reasoning_core.tree_of_thought import TreeOfThought
from src.reasoning_core.reflection import Reflection
from src.common.inference_client import InferenceClient

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """
    Specialized agent that performs reasoning using Chain‑of‑Thought,
    Tree‑of‑Thought, or Reflection strategies.
    """

    def __init__(
        self,
        agent_id: str = None,
        name: str = "ReasoningAgent",
        description: str = "Performs multi‑strategy reasoning",
        config: Optional[DOCAConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.REASONING,
            name=name,
            description=description,
            config=config,
        )
        self.inference_client = InferenceClient(self.config)
        self.reasoning_strategy = self.config.reasoning_core.default_model  # placeholder

    async def _execute(self, task: Task) -> ReasoningResult:
        """
        Execute a reasoning task using the appropriate strategy.
        The task input should contain at least 'prompt'.
        Optionally, 'max_tokens', 'temperature', and 'strategy' can be specified.
        """
        input_data = task.input_data
        prompt = input_data.get("prompt")
        if not prompt:
            raise ValueError("Task input missing 'prompt'")

        max_tokens = input_data.get("max_tokens", self.config.reasoning_core.max_tokens)
        temperature = input_data.get("temperature", self.config.reasoning_core.temperature)
        strategy = input_data.get("strategy", "cot")  # default: CoT

        self.logger.info(f"Running reasoning with strategy '{strategy}' on prompt: {prompt[:50]}...")

        result: ReasoningResult

        if strategy == "cot":
            cot = ChainOfThought(self.inference_client, self.config)
            result = await cot.run(prompt, max_tokens, temperature)
        elif strategy == "tot":
            tot = TreeOfThought(self.inference_client, self.config)
            result = await tot.run(
                prompt,
                max_tokens,
                temperature,
                num_branches=input_data.get("num_branches", 3),
                depth=input_data.get("depth", 1),
            )
        elif strategy == "reflection":
            ref = Reflection(self.inference_client, self.config)
            result = await ref.run(prompt, max_tokens, temperature)
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', falling back to CoT")
            cot = ChainOfThought(self.inference_client, self.config)
            result = await cot.run(prompt, max_tokens, temperature)

        self._log_result(result)
        return result