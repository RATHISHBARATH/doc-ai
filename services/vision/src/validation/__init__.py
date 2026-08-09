# ============================================================
# DOC AI Vision Service – Validation Package
# ============================================================

"""
Data validation and deduplication for ingested datasets.

This package provides:
- Deduplicator: Removes duplicate items based on hash or metadata.
- Verifier: Checks image integrity, metadata consistency, and source trust.
"""

from .deduplicator import Deduplicator
from .verifier import Verifier

__all__ = [
    "Deduplicator",
    "Verifier",
]