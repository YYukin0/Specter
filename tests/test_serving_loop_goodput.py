"""P7 Track C -- SpecServer with the goodput speculation-length controller.

Hermetic (make_fake_pair + LengthOnlyCache, no model loads). Pins:
  * controller="goodput" drains to completion and, at temperature 0 where
    speculative decoding is output-equivalent for any k, emits exactly what
    controller="fixed" emits;
  * a round-time model dominated by the draft/verify k-terms drives k* to 0
    (degraded rounds), visible as RoundInfo.controller_k == 0;
  * RoundInfo.controller_k is populated for the goodput controller and left at
    -1 otherwise.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goodput_model import RoundTimeCoeffs  # noqa: E402
from serving_loop import ServeConfig, SpecServer  # noqa: E402
from spec_oracles import LengthOnlyCache, _first_divergence, make_fake_pair  # noqa: E402

PROMPTS = ["hello world", "speculative decode", "a longer prompt here", "kv",
           "the quick brown fox", "one two three four"]

_COEFFS = RoundTimeCoeffs(c0=1e-3, c1=1e-4, c2=1e-6, c3=5e-4)


def _server(coeffs, controller, **over):
    draft, target, tok = make_fake_pair()
    cfg = ServeConfig(gamma=3, temperature=0.0, max_new_tokens=32, max_active=3,
                      make_cache=LengthOnlyCache, breaker_on=False,
                      controller=controller, goodput_coeffs=coeffs,
                      controller_warmup=3, controller_hysteresis=2)
    for k, v in over.items():
        setattr(cfg, k, v)
    return SpecServer(draft, target, tok, cfg)


def test_goodput_controller_drains_and_matches_fixed():
    srv_g = _server(_COEFFS, "goodput")
    srv_f = _server(_COEFFS, "fixed")
    for i, p in enumerate(PROMPTS):
        srv_g.submit(p, req_id=f"q{i}", seed=1000 + i)
        srv_f.submit(p, req_id=f"q{i}", seed=1000 + i)
    srv_g.run_until_idle(max_rounds=2000)
    srv_f.run_until_idle(max_rounds=2000)

    # At temperature 0, speculative decoding is output-equivalent for any k, so
    # goodput (varying k*) and fixed (gamma=3) share the same token stream; only
    # the max_new_tokens tail overshoot can differ, and by at most k_max.
    assert len(srv_g.results()) == len(PROMPTS)
    for i in range(len(PROMPTS)):
        a = srv_g.poll(f"q{i}").token_ids
        b = srv_f.poll(f"q{i}").token_ids
        assert _first_divergence(a, b, tol_tail=8) is None, \
            f"goodput vs fixed diverged on prompt {i}"


def test_goodput_controller_shrinks_k_to_zero_under_expensive_model():
    # blow up the k-linear terms (verify + draft cost) by 1e3 -> k* collapses to 0
    hot = RoundTimeCoeffs(c0=1e-3, c1=1e-1, c2=1e-6, c3=5e-1)
    srv = _server(hot, "goodput", controller_hysteresis=2)
    for i, p in enumerate(PROMPTS):
        srv.submit(p, req_id=f"q{i}", seed=1000 + i)
    infos = srv.run_until_idle(max_rounds=2000)

    assert len(srv.results()) == len(PROMPTS)
    ks = [inf.controller_k for inf in infos if inf.mode != "idle"]
    assert any(k == 0 for k in ks), f"expected a k*=0 round, saw {sorted(set(ks))}"
    assert any(inf.mode == "degraded" for inf in infos)


def test_controller_k_populated_for_goodput_only():
    srv_g = _server(_COEFFS, "goodput")
    srv_f = _server(_COEFFS, "fixed")
    for i, p in enumerate(PROMPTS):
        srv_g.submit(p, req_id=f"q{i}", seed=1000 + i)
        srv_f.submit(p, req_id=f"q{i}", seed=1000 + i)
    ig = srv_g.run_until_idle(max_rounds=2000)
    iff = srv_f.run_until_idle(max_rounds=2000)

    assert all(inf.controller_k != -1 for inf in ig if inf.mode != "idle")
    assert all(inf.controller_k == -1 for inf in iff)
    # after warmup the controller actually moved k off its gamma seed at least once
    post_warmup = [inf.controller_k for inf in ig if inf.mode != "idle"]
    assert any(k != 3 for k in post_warmup), "controller never changed k from the seed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
