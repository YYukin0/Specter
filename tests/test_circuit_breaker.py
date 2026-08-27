"""
P5.3 unit tests -- run with `pytest tests/` or `python tests/test_circuit_breaker.py`.

Two layers:
  * `circuit_breaker_decide` / `simulate_decisions` -- the pure state machine.
    A synthetic batch-size trace is fed in and the degrade / periodic-probe /
    recovery-probe / restore rounds are asserted step by step. No models.
  * A FakeModel end-to-end smoke of `circuit_breaker_generate` (no real weights,
    same FakeModel style as tests/test_gammatune.py, with a `parameters()` that
    exposes a `.device`).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit_breaker import (  # noqa: E402
    CircuitBreakerConfig,
    CircuitBreakerState,
    advance_state,
    circuit_breaker_decide,
    circuit_breaker_generate,
    measure_switch_cost,
    simulate_decisions,
)

# small config so the trace stays short and the periodic probe is reachable
CFG = CircuitBreakerConfig(batch_threshold=8, reprobe_every=5, reprobe_gamma=3, spec_gamma=3)


# --------------------------------------------------------------------------- #
# Pure state machine
# --------------------------------------------------------------------------- #
def test_stays_spec_while_batch_low():
    modes = [d.mode for d in simulate_decisions(CFG, [1, 2, 4, 7, 0, 3])]
    assert modes == ["spec"] * 6
    assert not any(d.switched for d in simulate_decisions(CFG, [1, 2, 4, 7, 0, 3]))


def test_degrade_is_immediate_on_threshold():
    #        step: 0  1  2  3  4
    trace = [1, 1, 8, 8, 8]
    ds = simulate_decisions(CFG, trace)
    assert ds[0].mode == "spec" and ds[1].mode == "spec"
    assert ds[2].mode == "target"
    assert ds[2].switched and ds[2].direction == "spec->target"
    # no lag: the degrade happens on the same step the signal crosses
    assert ds[3].mode == "target" and not ds[3].switched


def test_periodic_reprobe_while_degraded():
    # degrade at step 2, then stay high; reprobe_every=5 -> first periodic probe
    # fires 5 steps after the degrade clock reset, i.e. step 7
    trace = [1, 1] + [8] * 12
    ds = simulate_decisions(CFG, trace)
    assert ds[2].mode == "target"          # degrade
    assert [d.mode for d in ds[3:7]] == ["target"] * 4
    assert ds[7].mode == "reprobe"         # periodic probe, still degraded
    assert not ds[7].switched              # a probe is not a mode switch
    assert [d.mode for d in ds[8:12]] == ["target"] * 4
    # reprobe log carries per-step info via the driver, not here; but the clock
    # must have reset -> next periodic probe at step 12
    assert ds[12].mode == "reprobe"


def test_recovery_probe_then_restore():
    #        step: 0 1 2 3 4 5 6 7  8  9
    trace = [1, 1, 8, 8, 8, 8, 1, 1, 1, 1]
    ds = simulate_decisions(CFG, trace)
    assert ds[2].mode == "target"                       # degrade
    assert ds[6].mode == "reprobe"                      # batch dropped -> recovery probe
    assert not ds[6].switched
    assert ds[7].mode == "spec"                         # restore speculation
    assert ds[7].switched and ds[7].direction == "target->spec"
    assert [d.mode for d in ds[8:]] == ["spec", "spec"]


def test_short_degrade_still_probes_before_restore():
    # degraded for a single round, then batch clears immediately: still must run
    # one recovery probe (never flip straight back without an observation)
    trace = [1, 8, 1, 1]
    ds = simulate_decisions(CFG, trace)
    assert ds[1].mode == "target"
    assert ds[2].mode == "reprobe"
    assert ds[3].mode == "spec" and ds[3].switched


def test_advance_state_is_pure():
    s0 = CircuitBreakerState()
    d = circuit_breaker_decide(s0, 99, 0, CFG)
    s1 = advance_state(s0, d, 0)
    assert s0.mode == "spec"          # unchanged
    assert s1.mode == "target"
    assert d.direction == "spec->target"


# --------------------------------------------------------------------------- #
# FakeModel end-to-end smoke
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """logits[0, p, :] is a big logit at argmax_at(p) for every position p."""

    def __init__(self, argmax_at, vocab=16):
        self.argmax_at = argmax_at
        self.vocab = vocab
        self.generation_config = type("g", (), {"eos_token_id": None})()

    def parameters(self):
        return iter([torch.zeros(1)])  # driver reads .device off this

    def __call__(self, input_ids):
        seqlen = input_ids.shape[1]
        logits = torch.full((1, seqlen, self.vocab), -10.0)
        for p in range(seqlen):
            logits[0, p, self.argmax_at(p)] = 10.0
        return _Out(logits)


class _Tok:
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def _seq_ctx(tokenizer, prompt, device, apply_chat_template):
    return torch.zeros((1, 3), dtype=torch.long)


def test_circuit_breaker_generate_smoke(monkeypatch):
    import circuit_breaker

    monkeypatch.setattr(circuit_breaker, "encode_prompt", _seq_ctx)
    # draft and target agree -> speculative rounds fully accept, deterministic
    draft = FakeModel(lambda p: 1)
    target = FakeModel(lambda p: 1)

    # low -> high -> low so the run exercises degrade, a switch, and restore
    trace = [1, 1, 1] + [32] * 8 + [1] * 20
    res = circuit_breaker_generate(
        ["prompt one", "prompt two"], draft, target, _Tok(),
        config=CFG, batch_size_trace=trace,
        max_new_tokens=12, temperature=0.0, seed=0, measured_c=1.3,
    )

    assert len(res.texts) == 2
    assert res.emitted_total > 0
    assert res.n_rounds == len(res.per_round_mode)
    assert res.spec_rounds + res.target_rounds + res.reprobe_rounds == res.n_rounds
    # the high segment must have forced at least one degraded round and one switch
    assert res.target_rounds > 0
    assert any(s["direction"] == "spec->target" for s in res.mode_switches)
    assert set(res.cost_units_total) == {"measured", "c4", "c7", "c10"}
    assert all(v > 0 for v in res.cost_units_total.values())
    # switch cost probe populated once a switch happened
    assert res.switch_cost_probe and res.switch_cost_probe["switch_cost_ms"] >= 0.0
    # every reprobe entry has an alpha in [0, 1]
    for entry in res.reprobe_log:
        assert 0.0 <= entry["alpha"] <= 1.0


def test_measure_switch_cost_shape():
    m = FakeModel(lambda p: 1)
    out = measure_switch_cost(m, m, torch.zeros((1, 5), dtype=torch.long), reps=3)
    assert out["prefix_len"] == 5
    assert out["switch_cost_ms"] >= 0.0
    assert "proxy" in out


# --------------------------------------------------------------------------- #
def _run_all():
    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, val in reversed(self._undo):
                setattr(obj, name, val)
            self._undo.clear()

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            mp = _MP()
            try:
                fn(mp)
            finally:
                mp.undo()
        else:
            fn()
        print(f"  ok  {name}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
