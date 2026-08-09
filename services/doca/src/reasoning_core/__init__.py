# ============================================================
# DOC AI DOCA Service – Reasoning Core Package
# ============================================================

"""
Reasoning Core – Implements various reasoning strategies.

The reasoning core is responsible for:
- Chain‑of‑Thought (CoT) reasoning with step‑by‑step traces.
- Tree‑of‑Thought (ToT) reasoning with multiple branches.
- Reflection and self‑verification.
- Confidence scoring and uncertainty handling.
"""

from .chain_of_thought import ChainOfThought
from .tree_of_thought import TreeOfThought
from .reflection import Reflection

__all__ = [
    "ChainOfThought",
    "TreeOfThought",
    "Reflection",
]