"""P7 Track C -- hermetic unit tests for the pure goodput model.

No torch, no model loads. Pins the three closed-form pieces (Leviathan
expected-accept, linear round-time, goodput argmax) and the controller's
hysteresis clamp.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goodput_model import (  # noqa: E402
    RoundTimeCoeffs,
    best_k,
    expected_accepted_tokens,
    expected_round_time,
    goodput,
)


# --------------------------------------------------------------------------- #
# expected_accepted_tokens -- Leviathan et al. 2023 closed form
# --------------------------------------------------------------------------- #
def test_expected_accept_alpha_zero():
    assert expected_accepted_tokens(0.0, 5) == 1.0


def test_expected_accept_alpha_one_is_k_plus_one():
    assert expected_accepted_tokens(1.0, 5) == 6.0


def test_expected_accept_k1_matches_hand_formula():
    # (1 - a^2)/(1 - a) == 1 + a
    a = 0.77
    assert abs(expected_accepted_tokens(a, 1) - (1.0 + a)) < 1e-12


def test_expected_accept_k_zero_is_one_for_any_alpha():
    for a in (0.0, 0.3, 0.77, 1.0):
        assert expected_accepted_tokens(a, 0) == 1.0
    assert expected_accepted_tokens(0.9, -3) == 1.0


def test_expected_accept_monotone_in_k():
    a = 0.8
    vals = [expected_accepted_tokens(a, k) for k in range(0, 10)]
    assert all(b >= x for x, b in zip(vals, vals[1:]))


def test_expected_accept_monotone_in_alpha():
    k = 4
    vals = [expected_accepted_tokens(a, k) for a in (0.0, 0.2, 0.5, 0.8, 0.95, 1.0)]
    assert all(b >= x - 1e-12 for x, b in zip(vals, vals[1:]))


def test_expected_accept_alpha_near_one_no_blowup():
    # 坑30: no 0/0 as alpha -> 1
    v = expected_accepted_tokens(1.0 - 1e-12, 6)
    assert 6.0 <= v <= 7.0 + 1e-6


# --------------------------------------------------------------------------- #
# expected_round_time -- linear model monotonicity
# --------------------------------------------------------------------------- #
_C = RoundTimeCoeffs(c0=1e-3, c1=1e-4, c2=1e-6, c3=5e-4)


def test_round_time_increases_in_k():
    t = [expected_round_time(_C, n_active=4, mean_pending=1.0, mean_kv_len=100.0, k=k)
         for k in range(0, 9)]
    assert all(b > x for x, b in zip(t, t[1:]))


def test_round_time_increases_in_n_active():
    t = [expected_round_time(_C, n_active=n, mean_pending=1.0, mean_kv_len=100.0, k=3)
         for n in (1, 2, 4, 8)]
    assert all(b > x for x, b in zip(t, t[1:]))


def test_round_time_increases_in_kv_len():
    t = [expected_round_time(_C, n_active=4, mean_pending=1.0, mean_kv_len=kv, k=3)
         for kv in (10.0, 100.0, 500.0)]
    assert all(b > x for x, b in zip(t, t[1:]))


# --------------------------------------------------------------------------- #
# best_k -- goodput argmax picks a sane speculation length
# --------------------------------------------------------------------------- #
def test_best_k_high_alpha_low_load_speculates():
    # cheap draft, cheap verify, high accept, single stream -> want a long window
    coeffs = RoundTimeCoeffs(c0=1e-3, c1=1e-5, c2=1e-7, c3=1e-5)
    k_star, scores = best_k(coeffs, alpha=0.9, n_active=1, mean_pending=1.0,
                            mean_kv_len=64.0, k_min=0, k_max=8)
    assert k_star >= 3
    assert set(scores) == set(range(0, 9))


def test_best_k_low_alpha_high_load_disables_spec():
    # expensive draft + expensive verify, low accept, wide batch -> k* == 0
    coeffs = RoundTimeCoeffs(c0=1e-3, c1=5e-3, c2=1e-5, c3=5e-2)
    k_star, _ = best_k(coeffs, alpha=0.4, n_active=8, mean_pending=1.0,
                       mean_kv_len=400.0, k_min=0, k_max=8)
    assert k_star == 0


def test_best_k_hysteresis_clamps_swing():
    # even if the raw argmax is 0, prev_k=8 + hysteresis=2 floors it at 6
    coeffs = RoundTimeCoeffs(c0=1e-3, c1=5e-3, c2=1e-5, c3=5e-2)
    k_star, _ = best_k(coeffs, alpha=0.4, n_active=8, mean_pending=1.0,
                       mean_kv_len=400.0, k_min=0, k_max=8,
                       prev_k=8, hysteresis=2)
    assert k_star >= 6


def test_best_k_hysteresis_clamps_upswing():
    coeffs = RoundTimeCoeffs(c0=1e-3, c1=1e-5, c2=1e-7, c3=1e-5)
    k_star, _ = best_k(coeffs, alpha=0.95, n_active=1, mean_pending=1.0,
                       mean_kv_len=64.0, k_min=0, k_max=8,
                       prev_k=1, hysteresis=2)
    assert k_star <= 3


def test_goodput_zero_when_time_nonpositive():
    bad = RoundTimeCoeffs(c0=0.0, c1=0.0, c2=0.0, c3=0.0)
    assert goodput(bad, alpha=0.8, n_active=1, mean_pending=1.0,
                   mean_kv_len=0.0, k=0) == 0.0


# --------------------------------------------------------------------------- #
# RoundTimeCoeffs (de)serialisation
# --------------------------------------------------------------------------- #
def test_coeffs_json_round_trip(tmp_path):
    c = RoundTimeCoeffs(c0=1.5e-3, c1=2.5e-4, c2=3.5e-6, c3=4.5e-4, r2=0.91, n_fit=57)
    p = tmp_path / "p7_0_goodput_profile.json"
    p.write_text(json.dumps({"coeffs": c.as_dict()}))
    back = RoundTimeCoeffs.from_json(str(p))
    assert back == c
