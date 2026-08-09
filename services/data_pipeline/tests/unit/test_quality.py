# ============================================================
# Unit Tests for Quality Scoring Module
# ============================================================

import pytest
from src.quality_filter.quality_score import compute_quality_score, length_scoring
from src.common.config import QualityFilterConfig


@pytest.fixture
def quality_config():
    return QualityFilterConfig(
        min_words=3,
        max_words=100,
        max_punctuation_ratio=0.3,
        min_stop_word_ratio=0.05,
    )


def test_length_scoring_within_range():
    assert length_scoring(50, 10, 100) == 1.0


def test_length_scoring_below_min():
    score = length_scoring(5, 10, 100)
    assert 0.0 < score < 1.0


def test_length_scoring_above_max():
    score = length_scoring(150, 10, 100)
    assert 0.0 < score < 1.0


def test_compute_quality_score_high_quality(quality_config):
    text = "This is a high quality document with proper length and good structure."
    score = compute_quality_score(text, quality_config)
    assert score > 0.5


def test_compute_quality_score_low_quality(quality_config):
    text = "Short"
    score = compute_quality_score(text, quality_config)
    assert score < 0.5