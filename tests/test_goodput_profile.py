"""P7 Track C -- hermetic checks for the goodput calibration's pure logic.

Drives the real `_run_cell` code path with make_fake_pair + LengthOnlyCache (no
model loads, no MPS) for two grid cells, then runs the least-squares fit and
asserts it returns four finite coefficients with the right design-matrix shape.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rejection_sampling as _rs  # noqa: E402
from goodput_profile import _design_matrix, _fit, _run_cell  # noqa: E402
from spec_oracles import LengthOnlyCache, make_fake_pair  # noqa: E402
import random  # noqa: E402


def _cells():
    draft, target, tok = make_fake_pair()
    eos_ids = _rs.collect_eos_ids(tok, target)
    rows = []
    for n_active, k in [(1, 0), (2, 3)]:
        cell = _run_cell(draft, target, tok, eos_ids, n_active=n_active,
                         regime="short", k=k, device=torch.device("cpu"),
                         make_cache=LengthOnlyCache, dtype=torch.long)
        rows.extend(cell)
    return rows


def test_run_cell_produces_samples():
    rows = _cells()
    assert len(rows) >= 2
    for (n, mp, kv, k, t) in rows:
        assert n in (1, 2)
        assert mp >= 0.0
        assert kv >= 0.0
        assert k in (0, 3)
        assert t >= 0.0


def test_design_matrix_shape():
    rows = _cells()
    X = _design_matrix(rows, "base")
    assert X.shape == (len(rows), 4)
    Xp = _design_matrix(rows, "plus_nactive")
    assert Xp.shape == (len(rows), 5)


def test_fit_returns_four_finite_coeffs():
    # synthetic rows with a known-ish linear signal so lstsq is well posed
    rng = random.Random(0)
    rows = []
    for n in (1, 2, 4, 8):
        for k in (0, 1, 2, 3, 5, 8):
            mp, kv = 1.0, 50.0 + 10 * k
            t = 1e-3 + 1e-4 * n * (mp + k) + 1e-6 * n * kv + 5e-4 * n * k
            t += rng.uniform(-1e-5, 1e-5)
            rows.append((n, mp, kv, k, t))
    beta, r2, mape, n_tr, n_ho = _fit(rows, "base", random.Random(1000))
    assert len(beta) == 4
    assert np.all(np.isfinite(beta))
    assert np.isfinite(r2) and np.isfinite(mape)
    assert n_tr + n_ho == len(rows)
