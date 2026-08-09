# ============================================================
# DOC AI DOCA Service – Retriever Agent
# ============================================================

import logging
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.common.models import AgentType, Task
from src.common.config import DOCAConfig
from src.common.memory_client import MemoryClient

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Specialized agent that retrieves context from memory systems.

    Given a query, the RetrieverAgent queries the vector store (Milvus),
    knowledge graph (Neo4j), and long‑term memory (PostgreSQL) to gather
    relevant information. The retrieved context can be used by other
    agents to ground their reasoning in facts.
    """

    def __init__(
        self,
        agent_id: str = None,
        name: str = "RetrieverAgent",
        description: str = "Retrieves context from vector, graph, and long‑term memory",
        config: Optional[DOCAConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RETRIEVER,
            name=name,
            description=description,
            config=config,
        )
        self.memory_client = MemoryClient(self.config)

    async def _execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute a retrieval task.

        The task input should contain:
        - 'query': the search query (string)
        - 'top_k': optional number of results to return (int, default 5)
        - 'include_vector': whether to include vector search results (bool, default True)
        - 'include_graph': whether to include graph query results (bool, default False)
        - 'include_long_term': whether to include long‑term memory results (bool, default True)

        Returns a dict with:
        - 'context': a list of retrieved text snippets or documents
        - 'sources': a list of source identifiers (e.g., vector IDs, graph nodes)
        """
        input_data = task.input_data
        query = input_data.get("query")
        top_k = input_data.get("top_k", self.config.memory.vector_retrieval_top_k)
        include_vector = input_data.get("include_vector", True)
        include_graph = input_data.get("include_graph", False)
        include_long_term = input_data.get("include_long_term", True)

        if not query:
            raise ValueError("Task input missing 'query'")

        self.logger.info(f"Retrieving context for query: {query[:50]}...")

        context = []
        sources = []

        # 1. Vector search (Milvus) – placeholder for now
        if include_vector:
            vector_results = self.memory_client.search_vector(
                query_vector=[0.0] * 384,  # dummy vector; Milvus integration pending
                top_k=top_k,
                collection="default",
            )
            if vector_results:
                for result in vector_results:
                    context.append(result.get("text", ""))
                    sources.append(f"vector:{result.get('id', 'unknown')}")

        # 2. Graph query (Neo4j) – placeholder
        if include_graph:
            graph_results = self.memory_client.query_graph(
                cypher="MATCH (n) RETURN n LIMIT 5"  # dummy query
            )
            if graph_results:
                for result in graph_results:
                    context.append(str(result))
                    sources.append("graph:node")

        # 3. Long‑term memory (PostgreSQL)
        if include_long_term:
            # Use the query as a key to look up stored context
            # For simplicity, we just look up by the query string in the 'default' namespace
            long_term_result = self.memory_client.get_long_term(
                key=query,
                namespace="default"
            )
            if long_term_result:
                context.append(str(long_term_result))
                sources.append("long_term:default")

        # If no context found, provide a fallback message
        if not context:
            self.logger.warning("No context found for query. Using fallback.")
            context = ["No relevant context found."]
            sources = ["fallback"]

        self.logger.info(f"Retrieved {len(context)} context items.")

        return {
            "context": context,
            "sources": sources,
            "query": query,
        }