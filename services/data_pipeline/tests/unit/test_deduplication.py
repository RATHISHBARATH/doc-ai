# ============================================================
# Unit Tests for Deduplication Module
# ============================================================

import pytest
from src.deduplication.minhash_lsh import _tokenize_shingles, compute_minhashes, find_duplicates


def test_tokenize_shingles():
    text = "hello"
    shingles = _tokenize_shingles(text, n=3)
    assert shingles == ["hel", "ell", "llo"]


def test_compute_minhashes():
    docs = ["hello world", "hello world"]
    minhashes = compute_minhashes(docs, num_perm=8)
    assert len(minhashes) == 2


def test_find_duplicates_no_duplicates():
    docs = ["apple", "banana"]
    minhashes = compute_minhashes(docs, num_perm=8)
    duplicates = find_duplicates(minhashes, threshold=0.9, num_perm=8)
    assert len(duplicates) == 0


def test_find_duplicates_exact_duplicate():
    docs = ["hello world", "hello world"]
    minhashes = compute_minhashes(docs, num_perm=8)
    duplicates = find_duplicates(minhashes, threshold=0.5, num_perm=8)
    # One of the two should be marked as duplicate
    assert len(duplicates) == 1