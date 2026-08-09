# ============================================================
# DOC AI DOCA Service – Agent Factory Package
# ============================================================

"""
Agent Factory – Dynamically instantiates and manages agents.

The Agent Factory is responsible for:
- Creating agent instances based on configuration and task requirements.
- Caching and reusing agent instances when possible.
- Providing a registry of available agent types and their capabilities.
- Monitoring agent health and restarting failed agents.
"""

from .factory import AgentFactory
from .registry import AgentRegistry

__all__ = [
    "AgentFactory",
    "AgentRegistry",
]