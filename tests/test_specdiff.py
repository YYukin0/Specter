"""Meta-tests for src/specdiff.py -- the rule-based differential debugger.

specdiff's job is to take a decoder that diverges from a trusted reference and
name *why*: upstream KV/position corruption, sampling-rejection math, control
desync, or backend nondeterminism. These tests check:

  (1) a clean decoder vs itself -> NO_DIVERGENCE (no false positive);
  (2) `bisect` localises to the first structurally-different round;
  (3) every fault operator that manifests is classified to the mechanism in
      OP_EXPECTED_MECHANISM -- and specdiff never commits to a *wrong* mechanism
      (a mild mutant that doesn't perturb the trace on a seed is allowed to come
      back NO_DIVERGENCE, it just may not be counted as a hit);
  (4) the operators whose faultlib patches are reached through an imported name
      (`collect_eos_ids`, the `Injection`-based ones) still take effect inside
      specdiff's own driver -- regression guard for the module-object routing;
  (5) `adjusted_no_renormalize` is invisible to specdiff (torch.multinomial
      renormalizes) -- same finding as the O1/O3 oracles;
  (6) `blind_hunt` never mislabels a manifested mutant.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import specdiff as sd  # noqa: E402
from specdiff import (  # noqa: E402
    CONTROL_DESYNC,
    NO_DIVERGENCE,
    SAMPLING_MATH,
    UPSTREAM_KV_POS,
    RoundState,
    Trace,
    bisect,
    blind_hunt,
    diagnose,
)

PROMPTS = ["a longer prompt here", "the quick brown fox", "one two three four",
           "hello world"]
SEEDS = (1, 2, 3)
GAMMAS = (2, 3, 5)


# --------------------------------------------------------------------------- #
# no false positives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompt", PROMPTS)
def test_clean_decoder_has_no_divergence(prompt):
    for g in GAMMAS:
        rep = diagnose(prompt, (), gamma=g, seed=0)
        assert rep.mechanism == NO_DIVERGENCE
        assert not rep.diverged
        assert rep.offending_round is None


# --------------------------------------------------------------------------- #
# bisect
# --------------------------------------------------------------------------- #
def _round(idx, **over):
    base = dict(
        idx=idx, draft_tokens=[1, 2], accept_reject=(1, 1), n_accepted=2,
        emitted=(1, 2), draft_cache_len=10 + idx, target_cache_len=10 + idx,
        pos_ids_first=idx, cache_pos_calls=((idx,),),
        cached_verify_argmax0=3, recompute_argmax0=3, prefix_hash=f"h{idx}",
    )
    base.update(over)
    return RoundState(**base)


def test_bisect_finds_first_differing_round():
    ref = Trace(rounds=[_round(i) for i in range(5)])
    sus = Trace(rounds=[_round(i) for i in range(5)])
    assert bisect(ref, sus) is None

    sus.rounds[3] = _round(3, accept_reject=(1, 0), n_accepted=1)
    assert bisect(ref, sus) == 3


def test_bisect_flags_round_count_mismatch():
    ref = Trace(rounds=[_round(i) for i in range(5)])
    sus = Trace(rounds=[_round(i) for i in range(3)])
    assert bisect(ref, sus) == 3


# --------------------------------------------------------------------------- #
# per-operator mechanism classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(sd.OP_EXPECTED_MECHANISM))
def test_operator_classified_to_expected_mechanism(name):
    expected = sd.OP_EXPECTED_MECHANISM[name]
    seen = []
    for i, prompt in enumerate(PROMPTS):
        for g in GAMMAS:
            seen.append(diagnose(prompt, (name,), gamma=g, seed=i * 5 + g).mechanism)

    wrong = [m for m in seen if m not in (expected, NO_DIVERGENCE)]
    assert not wrong, f"{name}: specdiff named a wrong mechanism {set(wrong)} (expected {expected})"

    if expected != NO_DIVERGENCE:
        assert expected in seen, f"{name}: never manifested as {expected} across {len(seen)} runs"


# --------------------------------------------------------------------------- #
# module-object routing regression guards
# --------------------------------------------------------------------------- #
def test_eos_operator_takes_effect_inside_specdiff_driver():
    """collect_eos_ids is imported into specdiff; it must be reached through the
    module object so faultlib's eos_ignored_midblock patch fires."""
    got = [diagnose(p, ("eos_ignored_midblock",), gamma=3, seed=s).mechanism
           for p in PROMPTS for s in SEEDS]
    assert CONTROL_DESYNC in got
    assert all(m in (CONTROL_DESYNC, NO_DIVERGENCE) for m in got)


def test_injection_operator_takes_effect_inside_specdiff_driver():
    """force_accept_first works through rejection_sampling.Injection; the step
    function must be called via the module object for the wrap to apply."""
    got = [diagnose(p, ("force_accept_first",), gamma=3, seed=s).mechanism
           for p in PROMPTS for s in SEEDS]
    assert SAMPLING_MATH in got
    assert all(m in (SAMPLING_MATH, NO_DIVERGENCE) for m in got)


def test_kv_and_pos_operators_report_upstream():
    for name in ("kv_no_crop", "kv_crop_off_by_one_minus", "kv_crop_off_by_one_plus",
                 "pos_id_frozen", "pos_id_off_by_one_plus", "pos_id_off_by_one_minus"):
        got = [diagnose(p, (name,), gamma=g, seed=2).mechanism
               for p in PROMPTS for g in GAMMAS]
        assert UPSTREAM_KV_POS in got, f"{name} never reported UPSTREAM_KV_POS"
        assert all(m in (UPSTREAM_KV_POS, NO_DIVERGENCE) for m in got), (name, got)


def test_no_renormalize_is_invisible_to_specdiff():
    """torch.multinomial silently renormalizes -> identical samples -> identical
    trace. specdiff cannot and should not claim a mechanism here."""
    got = [diagnose(p, ("adjusted_no_renormalize",), gamma=g, seed=s).mechanism
           for p in PROMPTS for g in GAMMAS for s in SEEDS]
    assert set(got) == {NO_DIVERGENCE}


# --------------------------------------------------------------------------- #
# blind hunt
# --------------------------------------------------------------------------- #
def test_blind_hunt_never_mislabels_a_manifested_mutant():
    out = blind_hunt(n=24, gamma=3)
    assert out["n_manifested"] >= 12
    assert out["mechanism_precision_when_manifested"] == 1.0, out["confusion_expected_x_predicted"]
    # every non-NO_DIVERGENCE prediction sits on the diagonal
    for expected, preds in out["confusion_expected_x_predicted"].items():
        for got, k in preds.items():
            assert got in (expected, NO_DIVERGENCE), (expected, got, k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
