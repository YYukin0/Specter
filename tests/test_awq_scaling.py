"""Tests for P2.1 per-channel AWQ scaling search (src/awq_scaling.py).

Covers: fake-quant order/bounds, the scaling transform's mathematical identity,
the appendix-A.3 two-channel toy, and the alpha=0 / alpha=1 extremes.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from awq_scaling import (  # noqa: E402
    compute_scale,
    fake_quantize_groupwise,
    quant_grid_step,
    search_scale,
)


# --------------------------------------------------------------------------- #
# fake-quant
# --------------------------------------------------------------------------- #
def test_fake_quant_matches_manual_order():
    w = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    dq = fake_quantize_groupwise(w, n_bits=4, group_size=4)
    # scale = (100-1)/15 = 6.6 ; zero = round(-1/6.6) = 0
    # q = round(w/6.6): [0, 0, 0, 15] ; clamp: same ; dq = (q-0)*6.6
    assert torch.allclose(dq, torch.tensor([[0.0, 0.0, 0.0, 99.0]]), atol=1e-4)


def test_clamp_is_after_round_not_before():
    """Symmetric group [-4, 4]: qmax=15, scale=8/15, zero=round(4/scale)=round(7.5)=8.
    round(4/scale)=round(7.5)=8, +zero=16 -> WITHOUT clamp dq=(16-8)*scale=4.267;
    WITH clamp-after-round dq=(15-8)*scale=3.733 < 4.  If someone reordered to
    clamp-before-round the 4.0 entry would not be pulled below 4."""
    w = torch.tensor([[-4.0, 0.0, 2.0, 4.0]])
    scale = 8.0 / 15.0
    dq = fake_quantize_groupwise(w, n_bits=4, group_size=4)
    hi = dq[0, -1].item()
    assert hi == pytest.approx((15 - 8) * scale, abs=1e-4)
    assert hi < 4.0
    assert hi != pytest.approx(4.0 * (16 - 8) / 8, abs=1e-4)  # the no-clamp value 4.267


def test_fake_quant_stays_on_grid_and_in_range():
    torch.manual_seed(0)
    w = torch.randn(8, 128) * 0.3
    dq = fake_quantize_groupwise(w, n_bits=4, group_size=128)
    wf = w.reshape(8, 1, 128)
    qmax = 15
    scale = (wf.amax(-1, keepdim=True) - wf.amin(-1, keepdim=True)).clamp(min=1e-5) / qmax
    zero = torch.round(-wf.amin(-1, keepdim=True) / scale)
    q = (dq.reshape(8, 1, 128) / scale) + zero
    assert torch.all(q >= -1e-3) and torch.all(q <= qmax + 1e-3)
    assert torch.allclose(q, q.round(), atol=1e-3)  # dequant is last -> lands on integers


def test_fake_quant_is_idempotent():
    torch.manual_seed(1)
    w = torch.randn(4, 128) * 0.5
    dq1 = fake_quantize_groupwise(w, group_size=128)
    dq2 = fake_quantize_groupwise(dq1, group_size=128)
    assert torch.allclose(dq1, dq2, atol=1e-5)


def test_fake_quant_error_within_group_step():
    torch.manual_seed(2)
    w = torch.randn(16, 128) * 0.2
    dq = fake_quantize_groupwise(w, n_bits=4, group_size=128)
    step = quant_grid_step(w, n_bits=4, group_size=128)  # (16,1,1)
    err = (dq - w).abs().reshape(16, 1, 128)
    assert torch.all(err <= step + 1e-4)


# --------------------------------------------------------------------------- #
# scaling transform
# --------------------------------------------------------------------------- #
def test_scaling_transform_is_mathematical_identity():
    torch.manual_seed(0)
    W = torch.randn(32, 64)
    X = torch.randn(200, 64)
    act = X.abs().mean(0)
    wscale = W.abs().amax(0)
    for alpha in (0.0, 0.3, 0.7, 1.0):
        s = compute_scale(act, wscale, alpha)
        lhs = (X / s) @ (W * s).T
        rhs = X @ W.T
        assert torch.allclose(lhs, rhs, atol=1e-4), alpha


def test_alpha_extremes_formula():
    act = torch.tensor([10.0, 1.0, 4.0])
    wscale = torch.tensor([2.0, 8.0, 1.0])
    # alpha = 1 -> s ∝ act ; alpha = 0 -> s ∝ 1/wscale  (both up to the /sqrt(max*min) renorm)
    s1 = compute_scale(act, wscale, 1.0)
    assert torch.allclose(s1 / s1[0], act / act[0], atol=1e-5)
    s0 = compute_scale(act, wscale, 0.0)
    assert torch.allclose(s0 / s0[0], (1.0 / wscale) / (1.0 / wscale[0]), atol=1e-5)


# --------------------------------------------------------------------------- #
# appendix A.3 toy + AWQ objective
# --------------------------------------------------------------------------- #
def test_appendix_a3_toy_identity():
    """|X1|=10, |X2|=1, W1=W2=0.5 : the scale transform leaves the product exact."""
    X = torch.tensor([[10.0, 1.0], [-10.0, -1.0], [10.0, -1.0]])
    W = torch.tensor([[0.5, 0.5]])
    act = X.abs().mean(0)  # [10, 1]
    s = compute_scale(act, W.abs().amax(0), 1.0)
    assert torch.allclose((X / s) @ (W * s).T, X @ W.T, atol=1e-5)
    # channel with the big activation gets scaled UP (protected)
    assert s[0] > s[1]


def test_activation_aware_scaling_reduces_output_error():
    """AWQ Section 3 claim: with one input channel carrying ~10x the activation,
    an activation-aware scale (alpha > 0) gives lower quantized-output MSE than no
    scaling. Group spans both channels so the scale actually redistributes levels."""
    torch.manual_seed(0)
    out_f = 64
    W = torch.randn(out_f, 2) * 0.1
    X = torch.stack([torch.randn(512) * 10.0, torch.randn(512) * 1.0], dim=1)
    act = X.abs().mean(0)

    r = search_scale(W, X, act, n_bits=4, group_size=2)
    assert r["best_alpha"] is not None and r["best_alpha"] > 0.0   # search uses activation info
    assert r["best_out_mse"] < r["baseline_out_mse_no_scaling"]
    assert r["improvement_vs_no_scaling"] > 0.0

    # and protecting the high-activation channel shrinks ITS effective weight error
    wscale = W.abs().amax(0)
    s = compute_scale(act, wscale, r["best_alpha"])
    W_naive = fake_quantize_groupwise(W, n_bits=4, group_size=2)
    W_awq = fake_quantize_groupwise(W * s, n_bits=4, group_size=2) / s
    err_naive_ch0 = (W_naive[:, 0] - W[:, 0]).abs().mean()
    err_awq_ch0 = (W_awq[:, 0] - W[:, 0]).abs().mean()
    assert err_awq_ch0 < err_naive_ch0


def test_search_returns_grid_and_picks_min():
    torch.manual_seed(3)
    W = torch.randn(16, 8) * 0.1
    X = torch.randn(128, 8)
    act = X.abs().mean(0)
    r = search_scale(W, X, act, n_bits=4, group_size=8)
    alphas = [row["alpha"] for row in r["per_alpha"]]
    assert alphas == [None] + [round(0.1 * i, 1) for i in range(11)]  # None == no-scaling candidate
    best_row = min(r["per_alpha"], key=lambda x: x["out_mse"])
    assert r["best_alpha"] == best_row["alpha"]
    assert r["best_out_mse"] == best_row["out_mse"]
    assert r["per_alpha"][0]["alpha"] is None  # no-scaling always evaluated first
    assert r["per_alpha"][0]["out_mse"] == r["baseline_out_mse_no_scaling"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
