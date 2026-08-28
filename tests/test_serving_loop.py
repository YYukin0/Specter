"""P6.1 -- SpecServer: continuous batching + real-signal circuit breaker.

  (1) with the breaker off, every completed request equals its standalone
      `speculative_generate_kv` run (continuous batching and slot reuse do not
      perturb output);
  (2) requests submitted past `max_active` wait in the queue and are admitted as
      slots free up; all eventually complete;
  (3) the breaker degrades on a low rolling alpha, runs periodic probes while
      degraded, and restores speculation when alpha recovers -- and `len(active)`
      alone never trips it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from serving_loop import ServeConfig, SpecServer  # noqa: E402
from spec_kv import speculative_generate_kv  # noqa: E402
from spec_kv_batch import assert_rectangular_invariant  # noqa: E402
from spec_oracles import LengthOnlyCache, make_fake_pair  # noqa: E402

PROMPTS = ["hello world", "speculative decode", "a longer prompt here", "kv",
           "the quick brown fox", "one two three four"]


def _server(cfg_over=None, **fake):
    draft, target, tok = make_fake_pair(**fake)
    cfg = ServeConfig(gamma=3, temperature=0.0, max_new_tokens=28, max_active=2,
                      make_cache=LengthOnlyCache, breaker_on=False)
    if cfg_over:
        for k, v in cfg_over.items():
            setattr(cfg, k, v)
    return SpecServer(draft, target, tok, cfg), draft, target, tok


# --------------------------------------------------------------------------- #
def test_continuous_batching_output_equivalence():
    srv, draft, target, tok = _server()
    for i, p in enumerate(PROMPTS):
        srv.submit(p, req_id=f"q{i}", seed=100 + i)
    srv.run_until_idle()

    assert len(srv.results()) == len(PROMPTS)
    for i, p in enumerate(PROMPTS):
        ref = speculative_generate_kv(p, draft, target, tok, make_cache=LengthOnlyCache,
                                      gamma=3, max_new_tokens=28, temperature=0.0,
                                      seed=100 + i)
        assert srv.poll(f"q{i}").token_ids == ref.token_ids, p


def test_queue_admits_as_slots_free():
    srv, *_ = _server({"max_active": 2})
    for i, p in enumerate(PROMPTS):
        srv.submit(p, req_id=f"q{i}", seed=i)

    # first step: only 2 admitted, 4 still queued
    info = srv.step()
    assert len(info.admitted) == 2
    assert info.n_queued == 4
    assert info.n_active == 2

    infos = [info] + srv.run_until_idle()
    admitted_total = sorted(r for inf in infos for r in inf.admitted)
    assert admitted_total == [f"q{i}" for i in range(len(PROMPTS))]
    # slots freed by finished requests are refilled on a later round (admit runs
    # at the top of step(), so the refill shows up the round after the finish)
    assert any(inf.index > 0 and inf.admitted for inf in infos)
    first_finish = next(inf.index for inf in infos if inf.finished)
    assert any(inf.index > first_finish and inf.admitted for inf in infos)
    assert all(inf.n_active <= 2 for inf in infos)
    assert all(srv.poll(f"q{i}") is not None for i in range(len(PROMPTS)))


def test_admission_never_exceeds_max_active():
    srv, *_ = _server({"max_active": 3})
    for i in range(8):
        srv.submit(PROMPTS[i % len(PROMPTS)], req_id=f"q{i}", seed=i)
    for _ in range(200):
        if not (srv.active or srv.pending):
            break
        info = srv.step()
        assert info.n_active <= 3
        assert_rectangular_invariant(srv.active)
    assert len(srv.results()) == 8


# --------------------------------------------------------------------------- #
# circuit breaker
# --------------------------------------------------------------------------- #
def test_breaker_degrades_on_low_alpha_not_on_batch_size():
    # high acceptance pair -> even a full batch never degrades
    srv, *_ = _server({"breaker_on": True, "max_active": 6, "max_new_tokens": 40,
                       "warmup_rounds": 3, "alpha_floor": 0.5},
                      phase_target=0.3)
    for i, p in enumerate(PROMPTS):
        srv.submit(p, req_id=f"q{i}", seed=i)
    infos = srv.run_until_idle()
    assert all(inf.mode in ("spec", "idle") for inf in infos), \
        "high-alpha batch degraded purely on size"
    assert max(inf.n_active for inf in infos) >= 4  # batch really was large

    # low acceptance pair -> degrades, probes, and every probe/degraded round
    # is a real alpha decision
    srv2, *_ = _server({"breaker_on": True, "max_active": 4, "max_new_tokens": 60,
                        "temperature": 1.0, "warmup_rounds": 3, "alpha_floor": 0.5,
                        "reprobe_every": 6},
                       phase_target=2.0)
    for i, p in enumerate(PROMPTS):
        srv2.submit(p, req_id=f"q{i}", seed=i)
    infos2 = srv2.run_until_idle()
    modes = [inf.mode for inf in infos2]
    assert "degraded" in modes
    assert "probe" in modes
    assert srv2._rolling_alpha() < 0.5


def test_breaker_off_means_always_spec():
    srv, *_ = _server({"breaker_on": False, "temperature": 1.0}, phase_target=2.4)
    for i, p in enumerate(PROMPTS):
        srv.submit(p, req_id=f"q{i}", seed=i)
    infos = srv.run_until_idle()
    assert all(inf.mode in ("spec", "idle") for inf in infos)


def test_results_are_stable_across_max_active():
    outs = []
    for ma in (1, 2, 4, 8):
        srv, *_ = _server({"max_active": ma})
        for i, p in enumerate(PROMPTS):
            srv.submit(p, req_id=f"q{i}", seed=100 + i)
        srv.run_until_idle()
        outs.append({rid: r.token_ids for rid, r in srv.results().items()})
    for o in outs[1:]:
        assert o == outs[0], "output changed with batch width"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
