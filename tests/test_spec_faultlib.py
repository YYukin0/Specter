"""Meta-tests for the P6.5 fault library + oracle stack.

Two things must hold for a mutation-adequacy result to mean anything:
  (1) every operator in spec_faultlib actually changes observable behaviour
      (no dead mutants) -- caught by at least one oracle, or explicitly listed
      as an equivalent mutant;
  (2) each oracle actually kills the bug class it is supposed to, and -- the
      citable part -- the KV-management operators are invisible to BOTH
      output-equivalence oracles (O1 greedy, O3 sampling) and show up only as
      structural-invariant violations (O4).

Also checks the monkeypatch context managers fully restore state.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rejection_sampling as rs  # noqa: E402
import spec_faultlib as fl  # noqa: E402
import spec_kv as kv  # noqa: E402
import spec_kv_batch as kvb  # noqa: E402
from spec_oracles import run_o1, run_o3, run_o4, run_o5  # noqa: E402

FAST = dict(gammas=(3,), seeds=(0, 1))
FAST_O3 = dict(gammas=(3,), seeds=(0, 1, 2))

# behavioural equivalent mutants: same runtime effect, distinguishable only by a
# deprecation path / future breakage, not by any oracle here.
EQUIVALENT = {"kv_crop_absolute_vs_relative"}


def _eos(name):
    return name == "eos_ignored_midblock"


# --------------------------------------------------------------------------- #
def test_clean_baseline_passes_all_oracles():
    assert not run_o1(**FAST).killed
    assert not run_o3(**FAST_O3).killed
    assert not run_o4(**FAST).killed
    assert not run_o5(**FAST).killed


@pytest.mark.parametrize("name", fl.names())
def test_every_mutator_detected_by_some_oracle(name):
    e = _eos(name)
    killed = (
        run_o1((name,), eos=e, **FAST).killed
        or run_o3((name,), eos=e, **FAST_O3).killed
        or run_o4((name,), eos=e, **FAST).killed
    )
    if name in EQUIVALENT:
        assert not killed, f"{name} was expected to be an equivalent mutant"
    else:
        assert killed, f"{name} survives every oracle -- dead mutant or missing oracle"


# --------------------------------------------------------------------------- #
# oracle-class targeting: the headline of the mutation-adequacy matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [
    "kv_crop_off_by_one_minus", "kv_crop_off_by_one_plus", "kv_no_crop",
])
def test_kv_management_bugs_are_invisible_to_output_equivalence(name):
    """The citable finding: KV-cache management bugs pass BOTH greedy and
    sampling output-equivalence checks; only O4's structural invariants catch
    them. Ship parity tests only and these ship with you."""
    assert not run_o1((name,), **FAST).killed, f"{name} unexpectedly caught by O1"
    assert not run_o3((name,), **FAST_O3).killed, f"{name} unexpectedly caught by O3"
    assert run_o4((name,), **FAST).killed, f"{name} missed by O4"
    viol = run_o4((name,), **FAST).violations
    assert any("cache len" in v for v in viol), viol


@pytest.mark.parametrize("name", [
    "resample_from_target", "accept_strict", "leniency_injected", "bonus_token_from_draft",
])
def test_sampling_math_bugs_need_distributional_testing(name):
    """These rejection-sampling-math mutations are no-ops in greedy mode (every
    distribution is one-hot); O3's sampling-mode check is what catches them."""
    assert not run_o1((name,), eos=_eos(name), **FAST).killed
    assert run_o3((name,), eos=_eos(name), **FAST_O3).killed


def test_no_renormalize_only_caught_by_direct_assertion():
    """torch.multinomial silently renormalizes, so an unnormalized adjusted
    distribution is invisible to every output oracle -- only O4's explicit
    sum-to-one assertion (sampling-mode pass) catches it."""
    assert not run_o1(("adjusted_no_renormalize",), **FAST).killed
    assert not run_o3(("adjusted_no_renormalize",), **FAST_O3).killed
    r = run_o4(("adjusted_no_renormalize",), **FAST)
    assert r.killed
    assert any("sum=" in v for v in r.violations), r.violations


@pytest.mark.parametrize("name", ["pos_id_frozen", "pos_id_off_by_one_plus", "accept_ratio_inverted"])
def test_gross_bugs_caught_by_greedy_exact(name):
    assert run_o1((name,), **FAST).killed


# --------------------------------------------------------------------------- #
# O5 -- the batched decoder (P6.1) preserves single-sequence equivalence even
# with a fault operator active. The per-sequence-cache design has no shared
# ragged tensor, so batching cannot introduce a divergence class of its own.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", fl.names())
def test_batched_path_stays_equivalent_under_every_operator(name):
    r = run_o5((name,), eos=_eos(name), **FAST)
    assert not r.killed, (
        f"{name} broke spec_kv_batch.run_round vs single-sequence equivalence "
        f"({r.n_diverged}/{r.n_runs} runs) -- a real batch-path bug, not an "
        f"oracle catch"
    )


def test_faultlib_reaches_the_batched_step_binding():
    """`_inject`-style operators must patch spec_kv_batch's own
    `speculative_step_kv` binding, not just spec_kv's -- otherwise force-accept /
    bonus-provenance mutations silently no-op on the batched path."""
    import spec_kv_batch as _kvb
    orig = _kvb.speculative_step_kv
    with fl.apply("force_accept_first"):
        assert _kvb.speculative_step_kv is not orig
    assert _kvb.speculative_step_kv is orig


# --------------------------------------------------------------------------- #
# context-manager hygiene
# --------------------------------------------------------------------------- #
def test_apply_restores_all_patched_names():
    before = {
        ("rs", "acceptance_probability"): rs.acceptance_probability,
        ("kv", "acceptance_probability"): kv.acceptance_probability,
        ("rs", "adjusted_distribution"): rs.adjusted_distribution,
        ("kv", "adjusted_distribution"): kv.adjusted_distribution,
        ("kv", "_crop_to"): kv._crop_to,
        ("kv", "_cache_position"): kv._cache_position,
        ("rs", "collect_eos_ids"): rs.collect_eos_ids,
        ("rs", "speculative_step"): rs.speculative_step,
        ("kv", "speculative_step_kv"): kv.speculative_step_kv,
    }
    before[("kvb", "speculative_step_kv")] = kvb.speculative_step_kv
    before[("kvb", "dist_from_logits")] = kvb.dist_from_logits
    with fl.apply("accept_always", "kv_no_crop", "pos_id_frozen",
                  "eos_ignored_midblock", "bonus_token_from_draft"):
        pass
    after = {
        ("rs", "acceptance_probability"): rs.acceptance_probability,
        ("kv", "acceptance_probability"): kv.acceptance_probability,
        ("rs", "adjusted_distribution"): rs.adjusted_distribution,
        ("kv", "adjusted_distribution"): kv.adjusted_distribution,
        ("kv", "_crop_to"): kv._crop_to,
        ("kv", "_cache_position"): kv._cache_position,
        ("rs", "collect_eos_ids"): rs.collect_eos_ids,
        ("rs", "speculative_step"): rs.speculative_step,
        ("kv", "speculative_step_kv"): kv.speculative_step_kv,
        ("kvb", "speculative_step_kv"): kvb.speculative_step_kv,
        ("kvb", "dist_from_logits"): kvb.dist_from_logits,
    }
    assert before == after
    # and a clean oracle run still passes afterwards
    assert not run_o1(**FAST).killed


def test_unknown_mutator_raises():
    with pytest.raises(KeyError):
        with fl.apply("no_such_mutator"):
            pass


def test_deferred_operators_are_catalogued():
    # batch-only / step-rewrite operators are named with a reason, not silently
    # missing. After P6.1 the reason strings explain why each stays inactive
    # (per-seq caches make mask/pad drift inexpressible -- plan 坑19).
    assert "mask_left_pad_drift" in fl.DEFERRED
    assert "2510.22876" in fl.DEFERRED["mask_left_pad_drift"]
    assert all(isinstance(v, str) and v for v in fl.DEFERRED.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
