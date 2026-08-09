# ============================================================
# DOC AI DOCA Service – Workflow Scheduler
# ============================================================

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Awaitable

from src.common.models import Workflow, Task, TaskStatus
from src.common.config import DOCAConfig, get_config
from src.workflow_engine.dag_builder import DAGBuilder
from src.workflow_engine.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class WorkflowScheduler:
    """
    Schedules and orchestrates the execution of workflow DAGs.

    The WorkflowScheduler is responsible for:
    - Running workflows by executing tasks in topological order.
    - Managing task execution with retries and timeouts.
    - Tracking task states and updating the workflow state.
    - Handling failures and partial completions.

    It works with a TaskQueue (which abstracts NATS/Kafka) to enqueue and
    dequeue tasks for execution by agents.
    """

    def __init__(
        self,
        config: Optional[DOCAConfig] = None,
        task_queue: Optional[TaskQueue] = None,
        agent_executor: Optional[Callable[[Task], Awaitable[Task]]] = None,
    ):
        self.config = config or get_config()
        self.task_queue = task_queue or TaskQueue(self.config)
        self.agent_executor = agent_executor
        self.dag_builder = DAGBuilder()
        self.active_workflows: Dict[str, Workflow] = {}
        self.logger = logging.getLogger(f"{__name__}.WorkflowScheduler")

    async def submit_workflow(self, workflow: Workflow) -> str:
        """
        Submit a workflow for execution.

        Args:
            workflow: The workflow to execute (must have tasks and dependencies).

        Returns:
            The workflow ID.
        """
        # Validate the DAG
        self.dag_builder.build(workflow)

        # Store the workflow
        self.active_workflows[workflow.workflow_id] = workflow
        workflow.status = TaskStatus.RUNNING
        workflow.started_at = datetime.now()

        self.logger.info(f"Workflow {workflow.workflow_id} submitted with {len(workflow.tasks)} tasks")

        # Start execution (non-blocking)
        asyncio.create_task(self._execute_workflow(workflow))

        return workflow.workflow_id

    async def _execute_workflow(self, workflow: Workflow) -> None:
        """
        Execute a workflow by scheduling tasks in topological order.
        """
        # Get execution order (topological sort)
        try:
            execution_order = self.dag_builder.get_execution_order(workflow)
        except Exception as e:
            self.logger.error(f"Failed to compute execution order for workflow {workflow.workflow_id}: {e}")
            workflow.status = TaskStatus.FAILED
            workflow.error = str(e)
            return

        self.logger.info(f"Executing workflow {workflow.workflow_id} with {len(execution_order)} tasks")

        # Execute tasks in order
        for task_id in execution_order:
            task = workflow.tasks.get(task_id)
            if not task:
                self.logger.error(f"Task {task_id} not found in workflow")
                continue

            # Check if all dependencies are completed
            deps = workflow.dependencies.get(task_id, [])
            if deps:
                # Wait for dependencies to complete
                for dep_id in deps:
                    dep_task = workflow.tasks.get(dep_id)
                    if dep_task and dep_task.status != TaskStatus.COMPLETED:
                        self.logger.warning(f"Task {task_id} waiting for dependency {dep_id} to complete")
                        await asyncio.sleep(1)
                        # In a real implementation, we would use a more robust
                        # dependency tracking mechanism (e.g., callbacks or event listeners).

            # Execute the task
            await self._execute_task(task, workflow)

        # Check if all tasks are completed
        all_completed = all(
            t.status == TaskStatus.COMPLETED for t in workflow.tasks.values()
        )
        if all_completed:
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.now()
            self.logger.info(f"Workflow {workflow.workflow_id} completed successfully")
        else:
            # Some tasks failed or were skipped
            failed_tasks = [
                t.task_id for t in workflow.tasks.values()
                if t.status == TaskStatus.FAILED
            ]
            if failed_tasks:
                workflow.status = TaskStatus.FAILED
                workflow.error = f"Tasks failed: {failed_tasks}"
                self.logger.error(f"Workflow {workflow.workflow_id} failed: {failed_tasks}")
            else:
                # Some tasks are still pending (should not happen after full loop)
                workflow.status = TaskStatus.PENDING
                self.logger.warning(f"Workflow {workflow.workflow_id} incomplete after execution loop")

    async def _execute_task(self, task: Task, workflow: Workflow) -> None:
        """
        Execute a single task with retries and timeout handling.
        """
        max_retries = self.config.workflow_engine.max_retries
        retry_delay = self.config.workflow_engine.initial_delay_ms / 1000.0

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        attempt = 0
        while attempt <= max_retries:
            try:
                self.logger.info(f"Executing task {task.task_id} (attempt {attempt+1}/{max_retries+1})")

                # Execute the task using the agent executor
                if self.agent_executor:
                    executed_task = await self.agent_executor(task)
                    task.status = executed_task.status
                    task.result = executed_task.result
                    task.error = executed_task.error
                else:
                    # No executor provided – simulate execution
                    await asyncio.sleep(0.5)  # simulate work
                    task.status = TaskStatus.COMPLETED
                    task.result = {"message": "Task executed (simulated)"}

                if task.status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.now()
                    self.logger.info(f"Task {task.task_id} completed successfully")
                    return

                # If not completed and we have retries left, wait and retry
                if attempt < max_retries:
                    self.logger.warning(f"Task {task.task_id} failed (attempt {attempt+1}), retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.config.workflow_engine.backoff_factor
                    task.retries += 1

            except Exception as e:
                self.logger.error(f"Task {task.task_id} execution error: {e}")
                task.error = str(e)
                if attempt < max_retries:
                    self.logger.warning(f"Task {task.task_id} error, retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.config.workflow_engine.backoff_factor
                    task.retries += 1

            attempt += 1

        # All retries exhausted
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        self.logger.error(f"Task {task.task_id} failed after {max_retries+1} attempts")

    async def get_workflow_status(self, workflow_id: str) -> Optional[Workflow]:
        """
        Retrieve the current status of a workflow.
        """
        return self.active_workflows.get(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel an executing workflow.
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
        workflow.status = TaskStatus.CANCELLED
        self.logger.info(f"Workflow {workflow_id} cancelled")
        return True