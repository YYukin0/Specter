"""Tests for the full-model AWQ quantization pipeline (src/awq_quantize_model.py).

Strategy: a tiny hand-built module whose leaf Linear names match the target
suffixes, with calibration activations we control so we can force both a
"quantize this layer" outcome and a "no scaling wins -> fall back" outcome.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from awq_quantize_model import (  # noqa: E402
    capture_all_layer_inputs,
    quantize_model,
    summarize_records,
)
from awq_scaling import quant_grid_step  # noqa: E402


class TinyNet(nn.Module):
    """Two target Linears (q_proj: salient-channel case; down_proj: flat case) and
    one non-target (lm_head) that must be left alone."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(4, 6, bias=False)
        self.lm_head = nn.Linear(8, 5, bias=False)

    def forward(self, x):  # (b, t, 4)
        return self.lm_head(self.q_proj(x))


def _fixture():
    torch.manual_seed(0)
    net = TinyNet().eval()
    # q_proj: one input channel carries ~10x -> activation-aware scaling should help
    Xq = torch.stack([torch.randn(400) * 10.0, torch.randn(400) * 1.0,
                      torch.randn(400) * 1.0, torch.randn(400) * 1.0], dim=1)
    # down_proj: uniform activation magnitude AND uniform per-column weight max ->
    # compute_scale gives s == 1 for every alpha (renorm collapses it), so every
    # candidate ties the no-scaling baseline and the search falls back.
    Xd = torch.sign(torch.randn(400, 4)) * 2.0                        # abs mean exactly 2.0/ch
    Wd = (torch.randn(6, 4) * 0.3).clamp(-0.85, 0.85)
    Wd[0, :] = 0.9                                                    # every column max = 0.9
    net.down_proj.weight.data = Wd
    calib = {"q_proj": Xq, "down_proj": Xd}
    stats = {"abs_mean": {"q_proj": Xq.abs().mean(0), "down_proj": Xd.abs().mean(0)}}
    return net, stats, calib


def test_quantizes_target_layer_shape_and_dtype_preserved():
    net, stats, calib = _fixture()
    w_before = net.q_proj.weight.data.clone()
    lm_before = net.lm_head.weight.data.clone()

    recs = quantize_model(net, stats, calib, n_bits=4, group_size=4)

    assert recs["q_proj"]["fell_back"] is False
    assert not torch.equal(net.q_proj.weight.data, w_before)          # weight changed
    assert net.q_proj.weight.shape == w_before.shape
    assert net.q_proj.weight.dtype == w_before.dtype
    assert torch.equal(net.lm_head.weight.data, lm_before)            # non-target untouched


def test_quant_error_within_group_step_bound():
    net, stats, calib = _fixture()
    w_before = net.q_proj.weight.data.clone()
    recs = quantize_model(net, stats, calib, n_bits=4, group_size=4)
    # err vs the pre-quant weight, per output row, must sit inside that row's grid step
    # (the applied weight is fake_quant(W*s)/s, so bound is scaled -- use a loose
    # multiple of the unscaled step as a sanity ceiling)
    step = quant_grid_step(w_before, n_bits=4, group_size=4)          # (out,1,1)
    s_max = recs["q_proj"]["s_max"]
    s_min = recs["q_proj"]["s_min"]
    err = (net.q_proj.weight.data - w_before).abs().reshape(w_before.shape[0], 1, -1)
    assert torch.all(err <= step * (s_max / s_min) + 1e-3)


def test_no_scaling_case_falls_back_and_leaves_weight_bit_identical():
    net, stats, calib = _fixture()
    w_before = net.down_proj.weight.data.clone()
    recs = quantize_model(net, stats, calib, n_bits=4, group_size=4)

    assert recs["down_proj"]["fell_back"] is True
    assert recs["down_proj"]["alpha"] is None
    assert torch.equal(net.down_proj.weight.data, w_before)           # bit-identical
    assert recs["down_proj"]["out_mse_after"] == recs["down_proj"]["out_mse_before"]


def test_quantization_is_idempotent_with_frozen_scales():
    """The full pipeline is NOT idempotent on its own (the alpha search re-reads
    the now-quantized weights). Freezing the per-layer scale makes re-application
    a fixed point, because fake_quantize_groupwise itself is idempotent."""
    net, stats, calib = _fixture()
    sc = {}
    quantize_model(net, stats, calib, n_bits=4, group_size=4, scales_out=sc)
    w_once = net.q_proj.weight.data.clone()
    quantize_model(net, stats, calib, n_bits=4, group_size=4, frozen_scales=sc)
    assert torch.allclose(net.q_proj.weight.data, w_once, atol=1e-5)


def test_layers_limit_smoke_switch():
    net, stats, calib = _fixture()
    d_before = net.down_proj.weight.data.clone()
    recs = quantize_model(net, stats, calib, n_bits=4, group_size=4, layers_limit=1)
    # iter_target_linears yields q_proj first (module registration order) -> only it processed
    assert "q_proj" in recs and "down_proj" not in recs
    assert torch.equal(net.down_proj.weight.data, d_before)


def test_missing_calib_or_stat_marks_skipped():
    net, stats, calib = _fixture()
    del calib["down_proj"]
    recs = quantize_model(net, stats, calib, n_bits=4, group_size=4)
    assert recs["down_proj"]["skipped"] is True
    assert recs["down_proj"]["fell_back"] is False


def test_summarize_records_counts():
    net, stats, calib = _fixture()
    recs = quantize_model(net, stats, calib, n_bits=4, group_size=4)
    s = summarize_records(recs)
    assert s["n_target_layers"] == 2
    assert s["n_quantized"] + s["n_fell_back"] + s["n_skipped"] == 2
    assert s["n_quantized"] >= 1 and s["n_fell_back"] >= 1


def test_capture_all_layer_inputs_pools_and_caps():
    torch.manual_seed(1)
    net = TinyNet().eval()

    class FakeTok:
        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            n = 5 if text == "a" else 9
            return {"input_ids": torch.zeros(1, n, dtype=torch.long)}

    # model(input_ids=...) path -> adapt forward to accept the kwarg
    net.forward = lambda input_ids=None, **kw: net.lm_head(
        net.q_proj(torch.randn(1, input_ids.shape[1], 4)))
    got = capture_all_layer_inputs(net, FakeTok(), ["a", "b"],
                                   max_tokens_per_layer=10, max_seq_len=32)
    assert got["q_proj"].shape == (10, 4)          # 5 + 9 = 14 -> capped at 10
    assert "down_proj" not in got                  # never executed in this forward


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
