# ============================================================
# DOC AI DOCA Service – Planner Agent
# ============================================================

import logging
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.common.models import AgentType, Task, Workflow, TaskStatus
from src.common.config import DOCAConfig
from src.common.inference_client import InferenceClient

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Specialized agent that decomposes high‑level tasks into sub‑tasks
    and builds directed acyclic graphs (DAGs) for the Workflow Engine.

    Given a complex goal, the Planner breaks it down into smaller,
    executable steps, identifies dependencies between them, and
    produces a Workflow object ready for scheduling.
    """

    def __init__(
        self,
        agent_id: str = None,
        name: str = "PlannerAgent",
        description: str = "Decomposes tasks and builds workflow DAGs",
        config: Optional[DOCAConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.PLANNER,
            name=name,
            description=description,
            config=config,
        )
        self.inference_client = InferenceClient(self.config)

    async def _execute(self, task: Task) -> Workflow:
        """
        Execute a planning task.

        The task input should contain:
        - 'goal': the high‑level objective or problem statement (string)
        - 'context': optional context or constraints (dict)
        - 'max_steps': optional maximum number of sub‑tasks (int)

        Returns a Workflow object with tasks and dependencies populated.
        """
        input_data = task.input_data
        goal = input_data.get("goal")
        context = input_data.get("context", {})
        max_steps = input_data.get("max_steps", 8)

        if not goal:
            raise ValueError("Task input missing 'goal'")

        self.logger.info(f"Planning workflow for goal: {goal[:60]}...")

        # Build a prompt to guide the decomposition
        prompt = (
            f"Given the following goal, decompose it into a list of distinct sub‑tasks "
            f"that can be executed in parallel or sequentially. "
            f"Identify dependencies between tasks. "
            f"Return a structured list of tasks with brief descriptions and dependencies.\n\n"
            f"Goal: {goal}\n"
            f"Context: {context}\n"
            f"Maximum steps: {max_steps}\n\n"
            f"Format:\n"
            f"Task 1: description\n"
            f"Task 2: description (depends on Task 1)\n"
            f"Task 3: description (depends on Task 2)\n"
            f"..."
        )

        planning_text = self.inference_client.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.4,
        )

        if not planning_text:
            logger.warning("Planner: Decomposition failed. Creating a simple fallback workflow.")
            return self._fallback_workflow(goal)

        # Parse the generated text into tasks and dependencies
        tasks = []
        dependencies = {}

        lines = planning_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to match patterns like "Task 1: description" or "Task 1: description (depends on Task 2)"
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            task_label = parts[0].strip()
            rest = parts[1].strip()

            # Extract task description (and possibly dependency)
            dep_task_id = None
            if "(depends on" in rest.lower():
                desc_part, dep_part = rest.split("(depends on", 1)
                desc = desc_part.strip()
                dep_task_label = dep_part.strip(")").strip()
                # Map dependency label to task ID (simplified: we'll assign IDs later)
                dep_task_id = dep_task_label
            else:
                desc = rest

            # Assign a simplified task ID based on the label
            task_id = f"task_{len(tasks) + 1}"
            tasks.append((task_id, desc, dep_task_id))

        if not tasks:
            logger.warning("Planner: No valid tasks parsed. Using fallback.")
            return self._fallback_workflow(goal)

        # Build the Workflow object
        workflow = Workflow(
            name=f"Planned: {goal[:40]}...",
            description=goal,
        )

        # Create Task objects and dependency map
        task_objs = {}
        dep_map = {}
        for task_id, desc, dep_label in tasks:
            task_obj = Task(
                task_id=task_id,
                workflow_id=workflow.workflow_id,
                agent_type=AgentType.REASONING,  # default; can be overridden later
                input_data={"prompt": desc},
                status=TaskStatus.PENDING,
            )
            task_objs[task_id] = task_obj
            if dep_label:
                # Map dependency label to the actual task_id (simplified: we assume dep_label is a task label)
                # We'll just store the dependency as the label; we'll resolve later.
                dep_map[task_id] = [dep_label]

        # Populate the workflow
        workflow.tasks = task_objs
        workflow.dependencies = dep_map

        # Resolve dependencies (convert labels to actual task IDs)
        resolved_deps = {}
        for task_id, dep_labels in dep_map.items():
            resolved = []
            for label in dep_labels:
                # Try to find a task whose label matches the dependency
                for other_id, other_task in task_objs.items():
                    if other_id == label or f"task_{len(tasks)}" == label:  # naive matching
                        resolved.append(other_id)
                        break
                else:
                    # If no match, try to match by task order
                    if len(resolved) < len(tasks):
                        resolved.append(tasks[len(resolved)][0])
            resolved_deps[task_id] = resolved

        workflow.dependencies = resolved_deps

        self.logger.info(f"Planner: Created workflow with {len(workflow.tasks)} tasks.")
        return workflow

    def _fallback_workflow(self, goal: str) -> Workflow:
        """
        Create a simple fallback workflow when planning fails.
        """
        workflow = Workflow(
            name=f"Fallback: {goal[:40]}...",
            description=goal,
        )
        task = Task(
            task_id="task_1",
            workflow_id=workflow.workflow_id,
            agent_type=AgentType.REASONING,
            input_data={"prompt": f"Solve: {goal}"},
            status=TaskStatus.PENDING,
        )
        workflow.tasks = {"task_1": task}
        workflow.dependencies = {}
        return workflow