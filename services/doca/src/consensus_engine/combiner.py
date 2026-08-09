# ============================================================
# DOC AI DOCA Service – Consensus Combiner
# ============================================================

import logging
from typing import List, Dict, Any, Optional

from src.common.models import ReasoningResult, ConsensusResult, VotingMethod
from src.consensus_engine.voting import WeightedVoting
from src.consensus_engine.bayesian import BayesianCombination

logger = logging.getLogger(__name__)


class ConsensusCombiner:
    """
    Orchestrates the consensus aggregation process.

    The Combiner selects the appropriate aggregation method (weighted voting
    or Bayesian combination) based on the provided method parameter or
    a configured default, and returns a consolidated ConsensusResult.
    """

    def __init__(self, default_method: VotingMethod = VotingMethod.WEIGHTED):
        """
        Initialize the Combiner with a default consensus method.

        Args:
            default_method: The default voting method to use if not specified.
        """
        self.default_method = default_method
        self.logger = logging.getLogger(f"{__name__}.ConsensusCombiner")

    def combine(
        self,
        results: List[ReasoningResult],
        method: Optional[VotingMethod] = None,
        weights: Optional[Dict[str, float]] = None,
        priors: Optional[Dict[str, float]] = None,
    ) -> ConsensusResult:
        """
        Combine multiple reasoning results into a single consensus result.

        Args:
            results: List of ReasoningResult objects from different agents.
            method: The voting method to use. If None, the default is used.
            weights: Optional agent weights for weighted voting.
            priors: Optional prior probabilities for Bayesian combination.

        Returns:
            A ConsensusResult containing the final answer and confidence.
        """
        if not results:
            raise ValueError("No results to combine")

        # Use the specified method or fall back to default
        selected_method = method if method is not None else self.default_method
        self.logger.info(f"Combining {len(results)} results using {selected_method.value} method")

        if selected_method == VotingMethod.WEIGHTED:
            voting = WeightedVoting()
            return voting.aggregate(results, agent_weights=weights)

        elif selected_method == VotingMethod.BAYESIAN:
            bayesian = BayesianCombination()
            return bayesian.aggregate(results, prior_weights=priors)

        else:
            self.logger.warning(f"Unsupported method '{selected_method}', falling back to weighted voting")
            voting = WeightedVoting()
            return voting.aggregate(results, agent_weights=weights)