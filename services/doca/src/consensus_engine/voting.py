# ============================================================
# DOC AI DOCA Service – Weighted Voting Consensus
# ============================================================

import logging
from typing import List, Dict, Any, Optional

from src.common.models import ReasoningResult, ConsensusResult, VotingMethod

logger = logging.getLogger(__name__)


class WeightedVoting:
    """
    Implements weighted voting for consensus aggregation.

    Each agent's output is assigned a weight based on:
    - Historical accuracy (if available)
    - Confidence score of the agent's output
    - Agent type (some agents may be more reliable for certain tasks)

    The final answer is chosen as the one with the highest weighted sum.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WeightedVoting")

    def aggregate(
        self,
        results: List[ReasoningResult],
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> ConsensusResult:
        """
        Aggregate multiple reasoning results using weighted voting.

        Args:
            results: List of ReasoningResult objects from different agents.
            agent_weights: Optional mapping of agent_id -> weight.
                           If not provided, confidence scores are used as weights.

        Returns:
            ConsensusResult with the final answer, confidence, and metadata.
        """
        if not results:
            raise ValueError("No results to aggregate")

        # If only one result, return it directly
        if len(results) == 1:
            result = results[0]
            return ConsensusResult(
                final_answer=result.text,
                confidence=result.confidence,
                method=VotingMethod.WEIGHTED,
                agent_scores={},
                contributing_agents=[],
                metadata={"note": "Single agent, no aggregation needed"},
            )

        # Determine weights for each result
        # If agent_weights is provided, we use it. Otherwise, use confidence.
        weights = []
        for i, result in enumerate(results):
            if agent_weights and i < len(agent_weights):
                # Use provided weight (mapped by index or key)
                weight = list(agent_weights.values())[i] if isinstance(agent_weights, dict) else agent_weights[i]
            else:
                weight = result.confidence
            weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(results)] * len(results)

        # For text answers, we cannot average them. Instead, we:
        # 1. Group answers that are similar (simple approach: exact match)
        # 2. Sum weights for each unique answer
        # 3. Pick the answer with the highest total weight

        answer_weight_map: Dict[str, float] = {}
        for result, weight in zip(results, weights):
            text = result.text.strip()
            if text not in answer_weight_map:
                answer_weight_map[text] = 0.0
            answer_weight_map[text] += weight

        # Find the answer with the highest weight
        best_answer = max(answer_weight_map.items(), key=lambda x: x[1])
        final_answer = best_answer[0]
        final_confidence = best_answer[1]

        # Build metadata
        agent_scores = {}
        contributing_agents = []
        for i, result in enumerate(results):
            agent_id = f"agent_{i}"
            agent_scores[agent_id] = weights[i]
            contributing_agents.append(agent_id)

        self.logger.info(
            f"Weighted voting: selected answer with confidence {final_confidence:.2f} "
            f"from {len(results)} candidates"
        )

        return ConsensusResult(
            final_answer=final_answer,
            confidence=final_confidence,
            method=VotingMethod.WEIGHTED,
            agent_scores=agent_scores,
            contributing_agents=contributing_agents,
            metadata={
                "answer_weights": answer_weight_map,
                "num_candidates": len(results),
            },
        )