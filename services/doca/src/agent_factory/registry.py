# ============================================================
# DOC AI DOCA Service – Agent Registry
# ============================================================

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.common.models import AgentType

logger = logging.getLogger(__name__)


@dataclass
class AgentRegistration:
    """Registration record for an agent type."""
    agent_type: AgentType
    name: str
    description: str
    capabilities: List[str]
    default: bool = False
    enabled: bool = True
    metadata: Dict[str, Any] = None


class AgentRegistry:
    """
    Registry of available agent types and their capabilities.

    The AgentRegistry maintains a list of agent types that can be instantiated
    by the Agent Factory. It also stores metadata about each agent type,
    such as its capabilities and whether it is enabled by default.

    The registry can be populated from a configuration file (e.g., agents.yaml)
    or programmatically.
    """

    def __init__(self):
        self._registrations: Dict[AgentType, AgentRegistration] = {}
        self.logger = logging.getLogger(f"{__name__}.AgentRegistry")

    # ------------------------------------------------------------------
    # Registration management
    # ------------------------------------------------------------------

    def register(self, registration: AgentRegistration) -> None:
        """
        Register an agent type with the registry.

        Args:
            registration: The AgentRegistration record to add.
        """
        if registration.agent_type in self._registrations:
            self.logger.warning(f"Agent type {registration.agent_type} already registered. Overwriting.")
        self._registrations[registration.agent_type] = registration
        self.logger.info(f"Registered agent type: {registration.agent_type.value}")

    def register_from_config(self, config: Dict[str, Any]) -> None:
        """
        Register multiple agent types from a configuration dictionary.

        The expected format is:
        {
            "agents": [
                {
                    "id": "reasoning",
                    "type": "reasoning",
                    "description": "...",
                    "capabilities": ["...", "..."],
                    "default": true,
                    "enabled": true
                },
                ...
            ]
        }
        """
        agents_config = config.get("agents", [])
        for agent_cfg in agents_config:
            try:
                agent_type = AgentType(agent_cfg.get("type", "reasoning"))
                registration = AgentRegistration(
                    agent_type=agent_type,
                    name=agent_cfg.get("name", agent_type.value.capitalize()),
                    description=agent_cfg.get("description", ""),
                    capabilities=agent_cfg.get("capabilities", []),
                    default=agent_cfg.get("default", False),
                    enabled=agent_cfg.get("enabled", True),
                    metadata=agent_cfg.get("metadata", {}),
                )
                self.register(registration)
            except Exception as e:
                self.logger.error(f"Failed to register agent from config: {e}")

    def unregister(self, agent_type: AgentType) -> bool:
        """Remove an agent type from the registry."""
        if agent_type in self._registrations:
            del self._registrations[agent_type]
            self.logger.info(f"Unregistered agent type: {agent_type.value}")
            return True
        return False

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_registration(self, agent_type: AgentType) -> Optional[AgentRegistration]:
        """Get the registration record for a specific agent type."""
        return self._registrations.get(agent_type)

    def list_registrations(self) -> List[AgentRegistration]:
        """Return all registered agent types."""
        return list(self._registrations.values())

    def list_agent_types(self) -> List[AgentType]:
        """Return a list of all registered agent types."""
        return list(self._registrations.keys())

    def get_default_agent(self) -> Optional[AgentRegistration]:
        """Return the default agent type (if any)."""
        for reg in self._registrations.values():
            if reg.default and reg.enabled:
                return reg
        return None

    def get_agent_by_capability(self, capability: str) -> List[AgentRegistration]:
        """Return all agent types that have a specific capability."""
        results = []
        for reg in self._registrations.values():
            if reg.enabled and capability in reg.capabilities:
                results.append(reg)
        return results

    def is_registered(self, agent_type: AgentType) -> bool:
        """Check if an agent type is registered."""
        return agent_type in self._registrations

    def is_enabled(self, agent_type: AgentType) -> bool:
        """Check if an agent type is enabled."""
        reg = self._registrations.get(agent_type)
        return reg is not None and reg.enabled

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentRegistry":
        """
        Load the registry from a YAML configuration file.
        """
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        registry = cls()
        registry.register_from_config(config)
        return registry

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export the registry to a dictionary for serialization."""
        return {
            "agents": [
                {
                    "type": reg.agent_type.value,
                    "name": reg.name,
                    "description": reg.description,
                    "capabilities": reg.capabilities,
                    "default": reg.default,
                    "enabled": reg.enabled,
                    "metadata": reg.metadata,
                }
                for reg in self._registrations.values()
            ]
        }