# ============================================================
# DOC AI DOCA Service – Consensus Engine Package
# ============================================================

"""
Consensus Engine – Aggregates multiple agent outputs into a single result.

The consensus engine combines results from several agents (or multiple
reasoning paths) using weighted voting, Bayesian combination, or majority
voting to produce a final answer with a confidence score. This reduces
hallucinations and improves reliability.
"""

from .voting import WeightedVoting
from .bayesian import BayesianCombination
from .combiner import ConsensusCombiner

__all__ = [
    "WeightedVoting",
    "BayesianCombination",
    "ConsensusCombiner",
]