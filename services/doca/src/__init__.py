# ============================================================
# DOC AI DOCA Service – Package Initialization
# ============================================================

"""
DOCA Central Intelligence Service

This package provides the coordination, workflow, agent, reasoning,
and consensus components of the DOC AI Ecosystem's central intelligence.
"""

__version__ = "0.1.0"
__author__ = "DOC AI Ecosystem"

# Core modules
from . import common
from . import coordinator
from . import workflow_engine
from . import agent_factory
from . import reasoning_core
from . import consensus_engine
from . import agents

# Common exports
from .common.config import DOCAConfig, get_config, load_config
from .common.models import (
    Task, Agent, Workflow, ReasoningResult, ConsensusResult,
    TaskStatus, AgentType, VotingMethod
)
from .common.memory_client import MemoryClient
from .common.inference_client import InferenceClient

# Coordinator exports
from .coordinator.main import start_coordinator

# Workflow Engine exports
from .workflow_engine.dag_builder import DAGBuilder
from .workflow_engine.scheduler import WorkflowScheduler
from .workflow_engine.task_queue import TaskQueue

# Agent Factory exports
from .agent_factory.factory import AgentFactory
from .agent_factory.registry import AgentRegistry

# Reasoning Core exports
from .reasoning_core.chain_of_thought import ChainOfThought
from .reasoning_core.tree_of_thought import TreeOfThought
from .reasoning_core.reflection import Reflection

# Consensus Engine exports
from .consensus_engine.voting import WeightedVoting
from .consensus_engine.bayesian import BayesianCombination
from .consensus_engine.combiner import ConsensusCombiner

# Agent exports
from .agents.base_agent import BaseAgent
from .agents.reasoning_agent import ReasoningAgent
from .agents.reviewer_agent import ReviewerAgent
from .agents.planner_agent import PlannerAgent
from .agents.retriever_agent import RetrieverAgent

__all__ = [
    # Modules
    "common",
    "coordinator",
    "workflow_engine",
    "agent_factory",
    "reasoning_core",
    "consensus_engine",
    "agents",

    # Common
    "DOCAConfig",
    "get_config",
    "load_config",
    "Task",
    "Agent",
    "Workflow",
    "ReasoningResult",
    "ConsensusResult",
    "TaskStatus",
    "AgentType",
    "VotingMethod",
    "MemoryClient",
    "InferenceClient",

    # Coordinator
    "start_coordinator",

    # Workflow Engine
    "DAGBuilder",
    "WorkflowScheduler",
    "TaskQueue",

    # Agent Factory
    "AgentFactory",
    "AgentRegistry",

    # Reasoning Core
    "ChainOfThought",
    "TreeOfThought",
    "Reflection",

    # Consensus Engine
    "WeightedVoting",
    "BayesianCombination",
    "ConsensusCombiner",

    # Agents
    "BaseAgent",
    "ReasoningAgent",
    "ReviewerAgent",
    "PlannerAgent",
    "RetrieverAgent",
]