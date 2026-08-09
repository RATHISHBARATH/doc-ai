# ============================================================
# DOC AI DOCA Service – Task Dispatcher
# ============================================================

import asyncio
import logging
from typing import Dict, Optional, Any

from src.common.models import Task, Workflow, AgentType
from src.common.config import DOCAConfig, get_config
from src.workflow_engine.task_queue import TaskQueue
from src.agent_factory.factory import AgentFactory

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """
    Dispatches tasks to agents or enqueues them for asynchronous processing.

    The TaskDispatcher is responsible for:
    - Routing tasks to the appropriate agent (reasoning, reviewer, planner, etc.)
    - Calling the Agent Factory to spawn agents on demand.
    - Enqueuing tasks to the TaskQueue for background execution.
    - Providing synchronous execution for simple tasks.
    """

    def __init__(self, config: Optional[DOCAConfig] = None):
        self.config = config or get_config()
        self.task_queue = TaskQueue(self.config)
        self.agent_factory = AgentFactory(self.config)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(f"{__name__}.TaskDispatcher")

    async def start(self) -> None:
        """Start the dispatcher and connect to the task queue."""
        await self.task_queue.connect()
        self.logger.info("TaskDispatcher started")

    async def dispatch(self, task: Task) -> Task:
        """
        Dispatch a single task for processing.

        Depending on the agent type and task configuration, this method will:
        - Spawn a new agent via the Agent Factory (if needed).
        - Execute the task synchronously (if possible).
        - Or enqueue the task for asynchronous processing via NATS.

        Returns the updated task (with result or status).
        """
        self.logger.info(f"Dispatching task {task.task_id} to {task.agent_type.value}")

        # Determine if we should execute synchronously or enqueue.
        # For simplicity, we'll execute synchronously for now (since we have no
        # agent executors set up yet). In production, we'd use the task queue.

        if task.agent_type == AgentType.REASONING:
            from src.agents.reasoning_agent import ReasoningAgent
            agent = ReasoningAgent(config=self.config)
            return await agent.run(task)

        elif task.agent_type == AgentType.REVIEWER:
            from src.agents.reviewer_agent import ReviewerAgent
            agent = ReviewerAgent(config=self.config)
            return await agent.run(task)

        elif task.agent_type == AgentType.PLANNER:
            from src.agents.planner_agent import PlannerAgent
            agent = PlannerAgent(config=self.config)
            return await agent.run(task)

        elif task.agent_type == AgentType.RETRIEVER:
            from src.agents.retriever_agent import RetrieverAgent
            agent = RetrieverAgent(config=self.config)
            return await agent.run(task)

        else:
            self.logger.warning(f"Unknown agent type {task.agent_type}. Using fallback reasoning.")
            from src.agents.reasoning_agent import ReasoningAgent
            agent = ReasoningAgent(config=self.config)
            return await agent.run(task)

    async def dispatch_workflow(self, workflow: Workflow) -> None:
        """
        Dispatch all tasks in a workflow asynchronously.
        Each task is enqueued to the NATS task queue.
        """
        self.logger.info(f"Dispatching workflow {workflow.workflow_id} with {len(workflow.tasks)} tasks")

        for task_id, task in workflow.tasks.items():
            # Enqueue each task
            success = await self.task_queue.enqueue(task)
            if not success:
                self.logger.error(f"Failed to enqueue task {task_id}")

    async def shutdown(self) -> None:
        """Gracefully shut down the dispatcher."""
        await self.task_queue.close()
        self.logger.info("TaskDispatcher shutdown")