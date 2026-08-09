# ============================================================
# DOC AI DOCA Service – Test Package Initialization
# ============================================================

"""
Unit and integration tests for the DOCA service components.

The tests cover:
- Configuration loading and environment overrides.
- Core data models and serialization.
- Memory client operations (Redis, PostgreSQL).
- Inference client gRPC communication.
- Agent base class and specialized agents.
- Reasoning strategies (CoT, ToT, Reflection).
- Consensus engine (weighted voting, Bayesian).
- Workflow engine DAG building and scheduling.
- Task queue (NATS) integration.
- Agent factory and registry.
- Coordinator API endpoints.
"""