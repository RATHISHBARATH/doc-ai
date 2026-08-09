# ============================================================
# DOC AI DOCA Service – Coordinator API Router
# ============================================================
from datetime import datetime
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.common.models import TaskStatus, Workflow, AgentType
from src.common.config import get_config, DOCAConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------

class ReasonRequest(BaseModel):
    """Request to perform reasoning on a prompt."""
    prompt: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    use_review: bool = True
    use_retrieval: bool = True


class ReasonResponse(BaseModel):
    """Response containing the reasoned answer."""
    answer: str
    confidence: float
    reasoning_trace: Optional[list[str]] = None
    workflow_id: str


class TaskStatusResponse(BaseModel):
    """Response containing the status of a workflow."""
    workflow_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None


# ------------------------------------------------------------------
# In-memory workflow storage (temporary; replace with Redis/DB later)
# ------------------------------------------------------------------

# workflow_id -> Workflow
_workflows: dict[str, Workflow] = {}


# ------------------------------------------------------------------
# Dependency to get config
# ------------------------------------------------------------------

async def get_doca_config() -> DOCAConfig:
    return get_config()


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/reason", response_model=ReasonResponse)
async def reason(
    request: ReasonRequest,
    config: DOCAConfig = Depends(get_doca_config),
):
    """
    Submit a prompt for reasoning.
    This endpoint will create a workflow and start the reasoning process.
    """
    logger.info(f"Reason request received: {request.prompt[:50]}...")

    # Generate a workflow ID
    workflow_id = str(uuid4())

    # Create a workflow (simplified: we will just run a single reasoning task)
    workflow = Workflow(
        workflow_id=workflow_id,
        name=f"Reasoning on: {request.prompt[:30]}...",
        description=f"Prompt: {request.prompt}",
    )

    # Store the workflow
    _workflows[workflow_id] = workflow

    # TODO: Dispatch to actual workflow engine and agents
    # For now, we simulate a synchronous reasoning step.

    from src.reasoning_core.chain_of_thought import ChainOfThought
    from src.common.inference_client import InferenceClient

    # Initialize inference client
    inference_client = InferenceClient(config)

    # Run reasoning
    cot = ChainOfThought(inference_client, config)
    result = await cot.run(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    # Update workflow with result
    workflow.status = TaskStatus.COMPLETED
    workflow.final_result = result.text
    workflow.completed_at = datetime.now()

    return ReasonResponse(
        answer=result.text,
        confidence=result.confidence,
        reasoning_trace=result.reasoning_trace,
        workflow_id=workflow_id,
    )


@router.get("/workflow/{workflow_id}", response_model=TaskStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get the status and result of a workflow."""
    workflow = _workflows.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return TaskStatusResponse(
        workflow_id=workflow.workflow_id,
        status=workflow.status.value,
        created_at=workflow.created_at.isoformat(),
        completed_at=workflow.completed_at.isoformat() if workflow.completed_at else None,
        final_answer=workflow.final_result,
        error=workflow.error,
    )


@router.get("/workflows", response_model=list[TaskStatusResponse])
async def list_workflows(limit: int = 10):
    """List recent workflows."""
    workflows = list(_workflows.values())[-limit:]
    return [
        TaskStatusResponse(
            workflow_id=w.workflow_id,
            status=w.status.value,
            created_at=w.created_at.isoformat(),
            completed_at=w.completed_at.isoformat() if w.completed_at else None,
            final_answer=w.final_result,
            error=w.error,
        )
        for w in workflows
    ]