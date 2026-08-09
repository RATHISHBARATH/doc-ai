# ============================================================
# Unit Tests for Cleaning Module
# ============================================================

import pytest
from src.cleaning.clean_text import clean_document, scrub_pii
from src.common.config import CleaningConfig


@pytest.fixture
def cleaning_config():
    return CleaningConfig(
        fix_unicode=True,
        normalize_whitespace=True,
        remove_control_chars=True,
        filter_language="en",
        scrub_pii=True,
    )


def test_clean_document_basic(cleaning_config):
    text = "The quick brown fox  jumps   over the lazy dog."
    cleaned = clean_document(text, cleaning_config)
    assert cleaned == "The quick brown fox jumps over the lazy dog."


def test_clean_document_fix_unicode(cleaning_config):
    text = "Héllo wörld"
    cleaned = clean_document(text, cleaning_config)
    assert cleaned == "Héllo wörld"  # ftfy fixes but leaves valid chars


def test_clean_document_remove_control_chars(cleaning_config):
    text = "Hello\x00World"
    cleaned = clean_document(text, cleaning_config)
    assert cleaned == "HelloWorld"


def test_scrub_pii():
    text = "Contact me at john.doe@example.com or 555-123-4567."
    scrubbed = scrub_pii(text)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed


def test_clean_document_language_filter(cleaning_config):
    cleaning_config.filter_language = "fr"  # French
    text = "This is English text."
    cleaned = clean_document(text, cleaning_config)
    assert cleaned is None