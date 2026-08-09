# ============================================================
# Unit Tests for Tokenizer Training
# ============================================================

import pytest
from tokenizers import Tokenizer
from src.tokenizer_train.train import train_bpe


def test_train_bpe():
    texts = ["hello world", "world of AI", "hello AI"]
    tokenizer = train_bpe(texts, vocab_size=50, special_tokens=["[PAD]", "[UNK]"])
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.get_vocab_size() <= 50


def test_tokenizer_encodes_text():
    texts = ["hello world"]
    tokenizer = train_bpe(texts, vocab_size=50, special_tokens=["[PAD]", "[UNK]"])
    encoded = tokenizer.encode("hello world")
    assert len(encoded.ids) > 0