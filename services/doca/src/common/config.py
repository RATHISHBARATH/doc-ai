# ============================================================
# DOC AI DOCA Service – Configuration Loader
# ============================================================

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class CoordinatorConfig:
    """Configuration for the Coordinator service."""
    port: int = 50054
    http_port: int = 8001
    max_concurrent_workflows: int = 100


@dataclass
class WorkflowEngineConfig:
    """Configuration for the Workflow Engine."""
    max_parallel_tasks: int = 10
    default_timeout_seconds: int = 60
    max_retries: int = 3
    initial_delay_ms: int = 1000
    backoff_factor: float = 2.0


@dataclass
class AgentFactoryConfig:
    """Configuration for the Agent Factory."""
    agent_image: str = "doca-agent:latest"
    agent_memory_limit_mb: int = 512
    agent_timeout_seconds: int = 60


@dataclass
class ReasoningCoreConfig:
    """Configuration for the Reasoning Core."""
    default_model: str = "distilgpt2"
    adapter_version: str = "distilgpt2_lora_8_3epochs"
    max_tokens: int = 150
    temperature: float = 0.7


@dataclass
class ConsensusEngineConfig:
    """Configuration for the Consensus Engine."""
    num_agents: int = 3
    voting_method: str = "weighted"  # "weighted" | "bayesian"
    confidence_threshold: float = 0.7


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    short_term_ttl_seconds: int = 3600
    long_term_enabled: bool = True
    vector_retrieval_top_k: int = 5


@dataclass
class InferenceConfig:
    """Configuration for the inference client."""
    grpc_addr: str = "inference:50053"
    timeout_seconds: int = 10


@dataclass
class DOCAConfig:
    """Master configuration for the DOCA service."""
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    workflow_engine: WorkflowEngineConfig = field(default_factory=WorkflowEngineConfig)
    agent_factory: AgentFactoryConfig = field(default_factory=AgentFactoryConfig)
    reasoning_core: ReasoningCoreConfig = field(default_factory=ReasoningCoreConfig)
    consensus_engine: ConsensusEngineConfig = field(default_factory=ConsensusEngineConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: Path) -> "DOCAConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Override with environment variables (DOCA_* prefix)
        data = cls._apply_env_overrides(data)

        return cls(
            coordinator=CoordinatorConfig(**data.get("coordinator", {})),
            workflow_engine=WorkflowEngineConfig(**data.get("workflow_engine", {})),
            agent_factory=AgentFactoryConfig(**data.get("agent_factory", {})),
            reasoning_core=ReasoningCoreConfig(**data.get("reasoning_core", {})),
            consensus_engine=ConsensusEngineConfig(**data.get("consensus_engine", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            log_level=data.get("log_level", "INFO"),
        )

    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables (DOCA_*)."""
        # Simple implementation: flatten with double underscores for nesting.
        # Example: DOCA_COORDINATOR__PORT sets coordinator.port.
        for key, value in os.environ.items():
            if key.startswith("DOCA_"):
                # Remove prefix and split by '__'
                parts = key[5:].lower().split("__")
                target = data
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                # Convert value to appropriate type (string, int, float, bool)
                if value.lower() in ("true", "false"):
                    target[parts[-1]] = value.lower() == "true"
                elif value.isdigit():
                    target[parts[-1]] = int(value)
                else:
                    try:
                        target[parts[-1]] = float(value)
                    except ValueError:
                        target[parts[-1]] = value
        return data


# Singleton global config
_config: Optional[DOCAConfig] = None


def load_config(config_path: Optional[Path] = None) -> DOCAConfig:
    """Load the DOCA configuration from a YAML file."""
    if config_path is None:
        # Default path relative to the service root
        config_path = Path("/app/configs/doca.yaml")
    return DOCAConfig.from_yaml(config_path)


def get_config(config_path: Optional[Path] = None) -> DOCAConfig:
    """Get the global config instance (loads on first call)."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config():
    """Reset the singleton (useful for testing)."""
    global _config
    _config = None