# ============================================================
# DOC AI DOCA Service – Workflow Engine Package
# ============================================================

"""
Workflow Engine – Orchestrates the execution of task DAGs.

The Workflow Engine is responsible for:
- Building and validating Directed Acyclic Graphs (DAGs) of tasks.
- Scheduling tasks based on dependencies and available resources.
- Managing task queues (via NATS or Kafka) for reliable execution.
- Handling retries, timeouts, and task state persistence.
"""

from .dag_builder import DAGBuilder
from .scheduler import WorkflowScheduler
from .task_queue import TaskQueue

__all__ = [
    "DAGBuilder",
    "WorkflowScheduler",
    "TaskQueue",
]