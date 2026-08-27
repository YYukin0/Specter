"""Tests for the perplexity harness (src/awq_perplexity.py).

A constant-logits fake LM makes the expected perplexity exactly the vocab size,
so eval_perplexity's NLL math can be checked in closed form. Window/stride
bookkeeping is checked against hand counts.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from awq_perplexity import eval_perplexity  # noqa: E402


class ConstLM(nn.Module):
    """Emits uniform logits -> every token's NLL is exactly log(vocab)."""

    def __init__(self, vocab=97):
        super().__init__()
        self.vocab = vocab
        self._p = nn.Parameter(torch.zeros(1))  # gives parameters()/device

    def forward(self, input_ids=None, **_):
        b, L = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(b, L, self.vocab))


class FakeTok:
    def __init__(self, n_tokens):
        self.n = n_tokens

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        return {"input_ids": torch.arange(self.n).remainder(97).unsqueeze(0)}


def test_uniform_logits_perplexity_equals_vocab_size():
    m = ConstLM(vocab=97).eval()
    tok = FakeTok(2000)
    r = eval_perplexity(m, tok, ["x"], window=512, stride=512)
    assert r["perplexity"] == pytest.approx(97.0, rel=1e-4)
    assert r["mean_nll"] == pytest.approx(math.log(97), rel=1e-5)


def test_non_overlapping_window_counts():
    m = ConstLM().eval()
    tok = FakeTok(1000)
    r = eval_perplexity(m, tok, ["x"], window=250, stride=250)
    # begins 0,250,500,750 -> 4 windows; each scores L-1 = 249 -> 996
    assert r["n_windows"] == 4
    assert r["n_tokens"] == 4 * 249
    assert r["n_total_tokens"] == 1000


def test_overlapping_windows_do_not_double_count():
    m = ConstLM().eval()
    tok = FakeTok(1000)
    r = eval_perplexity(m, tok, ["x"], window=400, stride=200)
    # window 0 scores 399; each later window adds exactly `stride` (=200) new tokens
    # begins: 0,200,400,600 (end 1000 -> stop). windows = 4
    assert r["n_windows"] == 4
    assert r["n_tokens"] == 399 + 200 + 200 + 200
    # never more than the corpus
    assert r["n_tokens"] < r["n_total_tokens"]


def test_max_windows_caps_work():
    m = ConstLM().eval()
    tok = FakeTok(5000)
    r = eval_perplexity(m, tok, ["x"], window=256, stride=256, max_windows=3)
    assert r["n_windows"] == 3
    assert r["perplexity"] == pytest.approx(m.vocab, rel=1e-4)


def test_raises_when_corpus_shorter_than_window():
    m = ConstLM().eval()
    tok = FakeTok(100)
    with pytest.raises(RuntimeError):
        eval_perplexity(m, tok, ["x"], window=512, stride=512)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
