# ============================================================
# DOC AI DOCA Service – Base Agent
# ============================================================

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Union

from src.common.models import Agent, AgentType, Task, TaskStatus, ReasoningResult
from src.common.config import DOCAConfig, get_config

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all DOCA agents.

    Provides common functionality for agent lifecycle:
    - Agent identification (id, type, name, description)
    - Task execution and result handling
    - Logging and error handling
    - Heartbeat updates (if needed)
    """

    def __init__(
        self,
        agent_id: str = None,
        agent_type: AgentType = AgentType.REASONING,
        name: str = "BaseAgent",
        description: str = "",
        config: Optional[DOCAConfig] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.config = config or get_config()
        self._is_active = True
        self._current_task: Optional[Task] = None
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    # ------------------------------------------------------------------
    # Core lifecycle methods
    # ------------------------------------------------------------------

    async def run(self, task: Task) -> Task:
        """
        Execute the given task and return the updated task with results.
        Subclasses must override this method to provide actual logic.
        """
        self._current_task = task
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        try:
            self.logger.info(f"Agent {self.name} executing task {task.task_id}")
            result = await self._execute(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)

        self._current_task = None
        return task

    async def _execute(self, task: Task) -> Any:
        """
        Subclasses override this to implement their specific logic.
        Raises NotImplementedError by default.
        """
        raise NotImplementedError(f"Agent {self.name} must implement _execute()")

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def heartbeat(self) -> Dict[str, Any]:
        """Return agent status info for health monitoring."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "is_active": self._is_active,
            "current_task": self._current_task.task_id if self._current_task else None,
        }

    def stop(self) -> None:
        """Gracefully stop the agent (set inactive)."""
        self._is_active = False
        self.logger.info(f"Agent {self.name} stopped")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _format_result(self, text: str, confidence: float = 0.8, trace: Optional[list] = None) -> ReasoningResult:
        """Helper to create a standardized ReasoningResult."""
        return ReasoningResult(
            text=text,
            confidence=confidence,
            reasoning_trace=trace,
        )

    def _log_result(self, result: Any) -> None:
        """Log the result of a task at debug level."""
        if isinstance(result, ReasoningResult):
            self.logger.debug(f"Result: {result.text[:100]}... (conf={result.confidence})")
        else:
            self.logger.debug(f"Result: {str(result)[:100]}...")


# Import datetime for timestamp handling
from datetime import datetime