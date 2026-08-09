# ============================================================
# DOC AI DOCA Service – Coordinator Package
# ============================================================

"""
Coordinator Agent – The entry point for all DOCA tasks.

The Coordinator is responsible for:
- Receiving tasks from the gateway or other services.
- Deciding whether to handle a task directly or decompose it into a workflow.
- Dispatching tasks to the Workflow Engine or Agent Factory.
- Aggregating results and returning final responses.
"""

from .main import start_coordinator

__all__ = ["start_coordinator"]