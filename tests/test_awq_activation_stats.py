"""Tests for P2.0 activation-statistics collection (src/awq_activation_stats.py).

Strategy: a tiny hand-built module with target-suffix Linear names, fed known
inputs, so every per-channel number can be checked against a manual computation.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from awq_activation_stats import ActivationStatsCollector, collect, iter_target_linears  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 3, bias=False)      # target
        self.down_proj = nn.Linear(3, 2, bias=False)   # target
        self.lm_head = nn.Linear(2, 5, bias=False)     # NOT a target suffix

    def forward(self, x):
        return self.lm_head(self.down_proj(torch.relu(self.q_proj(x))))


def test_iter_target_linears_selects_by_leaf_name():
    block = TinyBlock()
    names = {n for n, _ in iter_target_linears(block)}
    assert names == {"q_proj", "down_proj"}
    assert "lm_head" not in names


def test_per_channel_abs_mean_matches_manual():
    torch.manual_seed(0)
    block = TinyBlock().eval()
    x = torch.randn(1, 6, 4)  # (batch, tokens, in_features) -> 6 tokens

    with torch.no_grad():
        c = ActivationStatsCollector(block)
        block(x)
        c.remove()
        res = c.result()

    # q_proj sees the block input directly
    expected_qmean = x.reshape(-1, 4).abs().mean(dim=0)
    expected_qmax = x.reshape(-1, 4).abs().amax(dim=0)
    assert torch.allclose(res["q_proj"]["abs_mean"], expected_qmean, atol=1e-6)
    assert torch.allclose(res["q_proj"]["abs_max"], expected_qmax, atol=1e-6)
    assert res["q_proj"]["n_tokens"] == 6

    # down_proj sees relu(q_proj(x))
    with torch.no_grad():
        h = torch.relu(block.q_proj(x)).reshape(-1, 3)
    assert torch.allclose(res["down_proj"]["abs_mean"], h.abs().mean(dim=0), atol=1e-6)
    assert torch.allclose(res["down_proj"]["abs_max"], h.abs().amax(dim=0), atol=1e-6)


def test_accumulates_across_multiple_forwards():
    block = TinyBlock().eval()
    x1 = torch.randn(1, 3, 4)
    x2 = torch.randn(1, 5, 4)

    with torch.no_grad():
        c = ActivationStatsCollector(block)
        block(x1)
        block(x2)
        c.remove()
        res = c.result()

    pooled = torch.cat([x1.reshape(-1, 4), x2.reshape(-1, 4)], dim=0)
    assert res["q_proj"]["n_tokens"] == 8
    assert torch.allclose(res["q_proj"]["abs_mean"], pooled.abs().mean(dim=0), atol=1e-6)
    assert torch.allclose(res["q_proj"]["abs_max"], pooled.abs().amax(dim=0), atol=1e-6)


def test_hooks_removed_after_context_exit():
    block = TinyBlock().eval()
    with ActivationStatsCollector(block) as c:
        assert len(c._handles) == 2
    assert c._handles == []
    # a forward after exit must not change anything
    before = {k: v.clone() for k, v in ((n, s["sum_abs"]) for n, s in c._acc.items() if s)}
    with torch.no_grad():
        block(torch.randn(1, 2, 4))
    for n, s in c._acc.items():
        if s is not None:
            assert torch.equal(s["sum_abs"], before[n])


def test_float32_accumulation_from_fp16_input():
    """Inputs in fp16 must be accumulated in fp32 (fp16 sum of many tokens loses
    precision / overflows). Check the running sum dtype and value."""
    block = TinyBlock().eval().half()
    x = (torch.randn(1, 10, 4) * 50).half()
    with torch.no_grad():
        c = ActivationStatsCollector(block)
        block(x)
        c.remove()
        res = c.result()
    assert res["q_proj"]["abs_mean"].dtype == torch.float32
    expected = x.reshape(-1, 4).to(torch.float32).abs().mean(dim=0)
    assert torch.allclose(res["q_proj"]["abs_mean"], expected, atol=1e-2)


def test_collect_reports_missing_layers():
    """A target layer on a branch that never runs should show up in missing()."""

    class Gated(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)  # never called

        def forward(self, x):
            return self.q_proj(x)

    class FakeTok:
        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    g = Gated().eval()
    # collect() calls model(input_ids=...) -> adapt Gated to accept the kwarg
    g.forward = lambda input_ids=None, **kw: g.q_proj(torch.randn(1, input_ids.shape[1], 4))  # type: ignore
    stats, meta = collect(g, FakeTok(), ["a", "b"], max_seq_len=8)
    assert "v_proj" in meta["missing_layers"]
    assert "q_proj" in stats


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
