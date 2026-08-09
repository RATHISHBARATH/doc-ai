# ============================================================
# DOC AI DOCA Service – Bayesian Combination Consensus
# ============================================================

import logging
import math
from typing import List, Dict, Any, Optional

from src.common.models import ReasoningResult, ConsensusResult, VotingMethod

logger = logging.getLogger(__name__)


class BayesianCombination:
    """
    Implements Bayesian combination for consensus aggregation.

    Each agent's output is treated as a probabilistic estimate. The Bayesian
    approach combines these estimates using prior beliefs and likelihoods
    to produce a posterior distribution over possible answers.

    This method is more robust than simple voting when agents have different
    reliability and when answers are not mutually exclusive.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BayesianCombination")

    def aggregate(
        self,
        results: List[ReasoningResult],
        prior_weights: Optional[Dict[str, float]] = None,
    ) -> ConsensusResult:
        """
        Aggregate multiple reasoning results using Bayesian combination.

        Args:
            results: List of ReasoningResult objects from different agents.
            prior_weights: Optional prior belief over each agent's reliability.
                           If not provided, confidence scores are used as the prior.

        Returns:
            ConsensusResult with the final answer, confidence, and metadata.
        """
        if not results:
            raise ValueError("No results to aggregate")

        if len(results) == 1:
            result = results[0]
            return ConsensusResult(
                final_answer=result.text,
                confidence=result.confidence,
                method=VotingMethod.BAYESIAN,
                agent_scores={},
                contributing_agents=[],
                metadata={"note": "Single agent, no aggregation needed"},
            )

        # Extract answers and confidences
        answers = [r.text.strip() for r in results]
        confidences = [r.confidence for r in results]

        # Compute priors for each answer (based on confidence)
        # We'll treat each agent's output as a hypothesis with prior = confidence
        # and likelihood = 1 - (1 - confidence) as a simple model.

        # Normalize confidences to use as priors
        total_conf = sum(confidences)
        if total_conf > 0:
            priors = [c / total_conf for c in confidences]
        else:
            priors = [1.0 / len(results)] * len(results)

        # Compute posterior probabilities using Bayes' theorem
        # For each answer, posterior = prior * likelihood / evidence
        # We'll use a simple model where likelihood is proportional to confidence.
        # More sophisticated models could incorporate answer quality scores.

        # For simplicity, we'll treat each agent's output as independent evidence.
        # The posterior for each answer is proportional to the product of priors
        # and the likelihood that the answer is correct given the agent's confidence.

        # We'll compute a score for each unique answer by combining the agents'
        # confidences in a Bayesian manner.
        answer_scores: Dict[str, float] = {}

        for answer, conf in zip(answers, confidences):
            if answer not in answer_scores:
                answer_scores[answer] = 1.0  # start with neutral prior

            # Bayesian update: posterior = prior * likelihood
            # We'll use conf as the likelihood that this answer is correct
            # and (1 - conf) as the likelihood that it's incorrect.
            # This is a simplified model; in practice, we'd use more sophisticated
            # likelihood functions.

            # Using odds form: Odds = (conf / (1 - conf))
            # Multiply odds for each agent that gave this answer
            odds = conf / (1 - conf + 1e-10)  # avoid division by zero
            answer_scores[answer] *= odds

        # Normalize scores to get a probability distribution
        total_score = sum(answer_scores.values())
        if total_score > 0:
            for ans in answer_scores:
                answer_scores[ans] /= total_score

        # Pick the answer with the highest posterior probability
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        final_answer = best_answer[0]
        final_confidence = best_answer[1]

        # Build metadata
        agent_scores = {}
        contributing_agents = []
        for i, result in enumerate(results):
            agent_id = f"agent_{i}"
            agent_scores[agent_id] = confidences[i]
            contributing_agents.append(agent_id)

        self.logger.info(
            f"Bayesian combination: posterior probability {final_confidence:.2f} "
            f"for answer '{final_answer[:50]}...' from {len(results)} agents"
        )

        return ConsensusResult(
            final_answer=final_answer,
            confidence=final_confidence,
            method=VotingMethod.BAYESIAN,
            agent_scores=agent_scores,
            contributing_agents=contributing_agents,
            metadata={
                "posterior_scores": answer_scores,
                "priors": priors,
                "num_candidates": len(results),
            },
        )