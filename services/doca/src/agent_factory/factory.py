# ============================================================
# DOC AI DOCA Service – Agent Factory
# ============================================================

import logging
from typing import Dict, Optional, Type, Any

from src.common.models import AgentType, Agent
from src.common.config import DOCAConfig, get_config
from src.agents.base_agent import BaseAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.retriever_agent import RetrieverAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Dynamically instantiates and manages agent instances.

    The Agent Factory maintains a registry of available agent types and
    can instantiate agents on demand. It also provides caching and
    lifecycle management for agents.
    """

    # Mapping from AgentType to the corresponding agent class
    _AGENT_CLASSES: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.REASONING: ReasoningAgent,
        AgentType.REVIEWER: ReviewerAgent,
        AgentType.PLANNER: PlannerAgent,
        AgentType.RETRIEVER: RetrieverAgent,
    }

    def __init__(self, config: Optional[DOCAConfig] = None):
        self.config = config or get_config()
        self._agent_cache: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger(f"{__name__}.AgentFactory")

    def create_agent(
        self,
        agent_type: AgentType,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> BaseAgent:
        """
        Create a new agent instance of the given type.

        Args:
            agent_type: The type of agent to create.
            name: Optional human‑readable name for the agent.
            description: Optional description of the agent's purpose.
            agent_id: Optional custom agent ID.

        Returns:
            An instance of the appropriate agent class.
        """
        agent_class = self._AGENT_CLASSES.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        # Use defaults if not provided
        name = name or f"{agent_type.value.capitalize()}Agent"
        description = description or f"Agent of type {agent_type.value}"

        instance = agent_class(
            agent_id=agent_id,
            name=name,
            description=description,
            config=self.config,
        )

        self.logger.info(f"Created agent {instance.agent_id} of type {agent_type.value}")
        return instance

    def get_or_create_agent(
        self,
        agent_type: AgentType,
        agent_id: Optional[str] = None,
        reuse_existing: bool = True,
    ) -> BaseAgent:
        """
        Get an existing agent from cache or create a new one.

        If reuse_existing is True and an agent with the same agent_id exists,
        it will be returned. Otherwise, a new agent is created.

        Args:
            agent_type: The type of agent to get or create.
            agent_id: The agent ID to look up or assign.
            reuse_existing: Whether to reuse cached agents.

        Returns:
            An agent instance.
        """
        if reuse_existing and agent_id and agent_id in self._agent_cache:
            self.logger.debug(f"Reusing cached agent {agent_id}")
            return self._agent_cache[agent_id]

        agent = self.create_agent(agent_type, agent_id=agent_id)
        # Cache the agent if it has a valid ID
        if agent.agent_id:
            self._agent_cache[agent.agent_id] = agent
        return agent

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a summary of all cached agents.
        """
        return {
            agent_id: {
                "type": agent.agent_type.value,
                "name": agent.name,
                "description": agent.description,
                "is_active": agent.is_active,
            }
            for agent_id, agent in self._agent_cache.items()
        }

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the cache (does not stop it).
        """
        if agent_id in self._agent_cache:
            del self._agent_cache[agent_id]
            self.logger.info(f"Removed agent {agent_id} from cache")
            return True
        return False

    def clear_cache(self) -> None:
        """Clear all cached agents."""
        self._agent_cache.clear()
        self.logger.info("Agent cache cleared")

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def stop_agent(self, agent_id: str) -> bool:
        """
        Gracefully stop a cached agent.
        """
        agent = self._agent_cache.get(agent_id)
        if agent:
            agent.stop()
            self.logger.info(f"Stopped agent {agent_id}")
            return True
        return False

    async def stop_all_agents(self) -> None:
        """Stop all cached agents."""
        for agent_id, agent in list(self._agent_cache.items()):
            agent.stop()
            self.logger.info(f"Stopped agent {agent_id}")