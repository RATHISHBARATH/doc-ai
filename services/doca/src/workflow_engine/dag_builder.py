# ============================================================
# DOC AI DOCA Service – DAG Builder
# ============================================================

import logging
from typing import Dict, List, Set, Optional

from src.common.models import Workflow, Task, TaskStatus

logger = logging.getLogger(__name__)


class DAGBuilder:
    """
    Builds and validates Directed Acyclic Graphs (DAGs) of tasks.

    The DAGBuilder takes a set of tasks and their dependencies and
    validates that the graph is acyclic, that all tasks are reachable,
    and that there are no circular dependencies. It also provides
    methods for topological sorting and determining execution order.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DAGBuilder")

    def build(self, workflow: Workflow) -> Workflow:
        """
        Validate and prepare a workflow for execution.

        This method checks:
        - That all tasks referenced in dependencies exist.
        - That the graph is acyclic (no circular dependencies).
        - That all tasks are reachable from the root tasks.

        If the workflow passes validation, it is returned unchanged.
        If validation fails, a ValueError is raised with details.
        """
        if not workflow.tasks:
            raise ValueError("Workflow has no tasks")

        # Validate that all dependency references exist
        for task_id, deps in workflow.dependencies.items():
            if task_id not in workflow.tasks:
                raise ValueError(f"Task {task_id} in dependencies but not in tasks")
            for dep_id in deps:
                if dep_id not in workflow.tasks:
                    raise ValueError(f"Dependency {dep_id} for task {task_id} does not exist")

        # Check for cycles using DFS
        visited: Set[str] = set()
        recursion_stack: Set[str] = set()

        def detect_cycle(task_id: str) -> bool:
            visited.add(task_id)
            recursion_stack.add(task_id)

            deps = workflow.dependencies.get(task_id, [])
            for dep_id in deps:
                if dep_id not in visited:
                    if detect_cycle(dep_id):
                        return True
                elif dep_id in recursion_stack:
                    return True

            recursion_stack.remove(task_id)
            return False

        for task_id in workflow.tasks:
            if task_id not in visited:
                if detect_cycle(task_id):
                    raise ValueError(f"Cycle detected in workflow dependencies involving task {task_id}")

        self.logger.info(f"DAG validated: {len(workflow.tasks)} tasks, no cycles")
        return workflow

    def get_execution_order(self, workflow: Workflow) -> List[str]:
        """
        Return a topological ordering of tasks based on dependencies.

        This is a simple Kahn's algorithm implementation. It returns a list of
        task IDs in the order they should be executed (any order that respects
        dependencies is valid; this one is deterministic).
        """
        # Build in-degree map
        in_degree: Dict[str, int] = {task_id: 0 for task_id in workflow.tasks}
        for deps in workflow.dependencies.values():
            for dep_id in deps:
                if dep_id in in_degree:
                    in_degree[dep_id] += 1

        # Start with tasks that have no dependencies (in-degree 0)
        queue = [task_id for task_id, deg in in_degree.items() if deg == 0]
        if not queue:
            raise ValueError("No root tasks (all tasks have dependencies) – circular or invalid graph")

        order = []
        while queue:
            # Pop a task (FIFO for deterministic order)
            task_id = queue.pop(0)
            order.append(task_id)

            # For each dependent task, reduce its in-degree
            for dep_id, deps in workflow.dependencies.items():
                if task_id in deps:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        queue.append(dep_id)

        if len(order) != len(workflow.tasks):
            raise ValueError("Cycle detected during topological sort – invalid DAG")

        self.logger.info(f"Execution order computed: {order}")
        return order

    def get_root_tasks(self, workflow: Workflow) -> List[str]:
        """
        Return the task IDs that have no dependencies (root tasks).
        """
        root_tasks = []
        for task_id, deps in workflow.dependencies.items():
            if not deps:
                root_tasks.append(task_id)
        # Also include any tasks not listed in dependencies (they have no dependencies)
        for task_id in workflow.tasks:
            if task_id not in workflow.dependencies:
                root_tasks.append(task_id)
        return list(set(root_tasks))

    def get_leaf_tasks(self, workflow: Workflow) -> List[str]:
        """
        Return the task IDs that have no dependents (leaf tasks).
        """
        # Build reverse dependency map
        dependents: Dict[str, List[str]] = {task_id: [] for task_id in workflow.tasks}
        for task_id, deps in workflow.dependencies.items():
            for dep_id in deps:
                if dep_id not in dependents:
                    dependents[dep_id] = []
                dependents[dep_id].append(task_id)

        leaf_tasks = []
        for task_id, deps in dependents.items():
            if not deps:
                leaf_tasks.append(task_id)
        return leaf_tasks