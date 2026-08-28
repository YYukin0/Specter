"""P6.7 -- fused Metal accept/reject kernel + roofline.

The byte/flop accounting and the pure-MLX reference helpers are hermetic and
always run. The two tests that launch the actual Metal kernel run only when a
Metal GPU is available, else skip.
"""
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import metal_accept_kernel as mak  # noqa: E402

_HAS_METAL = mx.metal.is_available() if hasattr(mx, "metal") else False
_metal = pytest.mark.skipif(not _HAS_METAL, reason="no Metal GPU")


# --------------------------------------------------------------------------- #
# hermetic: traffic model
# --------------------------------------------------------------------------- #
def test_fused_moves_less_than_naive():
    for g in (2, 4, 8):
        fused = mak.fused_traffic(g, mak.VOCAB_QWEN25)
        naive = mak.reference_branchless_traffic(g, mak.VOCAB_QWEN25)
        assert fused.bytes_moved < naive.bytes_moved


def test_traffic_scales_with_gamma_and_vocab():
    assert mak.fused_traffic(8, 1000).bytes_moved > mak.fused_traffic(2, 1000).bytes_moved
    assert mak.fused_traffic(4, 2000).bytes_moved > mak.fused_traffic(4, 1000).bytes_moved


def test_intensity_is_flops_over_bytes():
    t = mak.Traffic(bytes_moved=200, flops=100)
    assert t.intensity == 0.5


def test_op_is_memory_bound_on_this_class():
    # arith intensity for both impls must be << a plausible ridge point (>= 10)
    for g in (2, 4, 8):
        assert mak.fused_traffic(g, mak.VOCAB_QWEN25).intensity < 10
        assert mak.reference_branchless_traffic(g, mak.VOCAB_QWEN25).intensity < 10


# --------------------------------------------------------------------------- #
# hermetic: pure-MLX reference logic
# --------------------------------------------------------------------------- #
def test_accept_scan_leading_run():
    pt_x = mx.array([0.9, 0.9, 0.9, 0.9])
    pd_x = mx.array([0.1, 0.1, 0.1, 0.1])          # accept prob clamped to 1
    assert mak._accept_scan(pt_x, pd_x, mx.array([0.0, 0.0, 0.5, 0.0])) == 4
    # first reject stops the run even if later ones would accept
    pt_x2 = mx.array([1.0, 0.0, 1.0, 1.0])
    pd_x2 = mx.array([1.0, 1.0, 1.0, 1.0])
    assert mak._accept_scan(pt_x2, pd_x2, mx.array([0.5, 0.5, 0.0, 0.0])) == 1


def test_all_adjusted_rows_shape_and_normalisation():
    g, v = 3, 64
    tl = mx.random.normal((g + 1, v), key=mx.random.key(1))
    dl = mx.random.normal((g + 1, v), key=mx.random.key(2))
    pt = mx.softmax(tl, axis=-1)
    pd = mx.softmax(dl, axis=-1)
    rows = mak._all_adjusted_rows(pt, pd)
    assert rows.shape == (g + 1, v)
    sums = mx.sum(rows, axis=-1)
    assert bool(mx.all(mx.abs(sums - 1.0) < 1e-4))
    assert bool(mx.all(rows >= -1e-6))               # valid distribution


def test_reference_sync_matches_branchless():
    for g, af in ((3, 0.9), (5, 0.5), (7, 0.2)):
        tl, dl, toks, unif = mak.make_inputs(g, 512, seed=g, accept_frac=af)
        n1, a1 = mak.reference_sync(tl, dl, toks, unif)
        n2, a2 = mak.reference_branchless(tl, dl, toks, unif)
        assert n1 == n2
        assert float(mx.max(mx.abs(a1 - a2))) < 1e-5


# --------------------------------------------------------------------------- #
# Metal-gated: the kernel itself
# --------------------------------------------------------------------------- #
@_metal
def test_fused_matches_reference_small_vocab():
    for g, af in ((2, 0.9), (4, 0.6), (6, 0.3), (8, 0.95)):
        tl, dl, toks, unif = mak.make_inputs(g, 256, seed=g * 3 + 1, accept_frac=af)
        rn, ra = mak.reference_branchless(tl, dl, toks, unif)
        fn, fa = mak.fused_accept(tl, dl, toks, unif)
        assert rn == fn, (g, af, rn, fn)
        assert float(mx.max(mx.abs(fa - ra))) < 1e-5
        assert abs(float(mx.sum(fa)) - 1.0) < 1e-4


@_metal
def test_fused_bonus_path_all_accepted():
    # force all drafts accepted -> n == gamma -> adjusted row is the bonus (target) row
    g = 4
    tl, dl, toks, _ = mak.make_inputs(g, 256, seed=11, accept_frac=0.99)
    unif = mx.zeros((g,))                            # r = 0 -> always accept
    n, adj = mak.fused_accept(tl, dl, toks, unif)
    assert n == g
    pt_bonus = mx.softmax(tl[g].astype(mx.float32), axis=-1)
    assert float(mx.max(mx.abs(adj - pt_bonus))) < 1e-5
