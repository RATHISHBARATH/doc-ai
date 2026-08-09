# ============================================================
# DOC AI DOCA Service – Core Data Models
# ============================================================

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union


class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Types of agents available in the DOCA ecosystem."""
    REASONING = "reasoning"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    SYNTHESIZER = "synthesizer"
    CODE = "code"
    VISION = "vision"  # reserved for Phase 6


class VotingMethod(Enum):
    """Methods for consensus aggregation."""
    WEIGHTED = "weighted"
    BAYESIAN = "bayesian"
    MAJORITY = "majority"


@dataclass
class Task:
    """A single unit of work within a workflow."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    agent_type: AgentType = AgentType.REASONING
    input_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Represents an agent instance with its configuration and state."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.REASONING
    name: str = ""
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """A Directed Acyclic Graph (DAG) of tasks to be executed."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # task_id -> list of dependent task_ids
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Output from a reasoning agent, including confidence and trace."""
    text: str
    confidence: float
    reasoning_trace: Optional[List[str]] = None  # Chain-of-Thought steps
    alternatives: Optional[List[str]] = None    # For Tree-of-Thought
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Final aggregated output from the consensus engine."""
    final_answer: str
    confidence: float
    method: VotingMethod
    agent_scores: Dict[str, float] = field(default_factory=dict)  # agent_id -> weight
    contributing_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)