"""
P5.0 unit tests -- run with `pytest tests/` or `python tests/test_gammatune.py`.

`test_appendix_a2_three_round_example` reproduces the hand-worked table in
notes/project_plan_v9.md appendix A.2 and asserts the exact (gamma, gamma_bar)
after each round for A = [3, 2, 3]:

    round 1 (A==gamma, expand) -> (5, 3.0)
    round 2 (A<gamma,  EMA)    -> (3, 2.7)
    round 3 (A==gamma, expand) -> (5, 2.7)

This is the "known-correct" reference point that P5.1 and P5.4 build on. Float
comparisons use abs(...) < 1e-9.

The FakeModel test drives `gammatune_generate` end to end without real weights and
checks that the recorded gamma_trace actually tracks a changing accept pattern and
that `carry_state` seeds the controller instead of resetting it.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gammatune import (  # noqa: E402
    GammaTuneConfig,
    gammatune_generate,
    gammatune_update,
)


# --------------------------------------------------------------------------- #
# Appendix A.2 hand-worked three-round example
# --------------------------------------------------------------------------- #
def test_appendix_a2_three_round_example():
    eta, delta, gmin, gmax = 0.3, 2, 1, 10
    gamma, gamma_bar = 3, 3.0  # initial state from the appendix

    # round 1: A = 3 == gamma -> expand branch, gamma <- 3 + 2, gamma_bar unchanged
    gamma, gamma_bar = gammatune_update(gamma, gamma_bar, 3, eta=eta, delta=delta, gmin=gmin, gmax=gmax)
    assert gamma == 5, gamma
    assert abs(gamma_bar - 3.0) < 1e-9, gamma_bar

    # round 2: A = 2 < gamma(5) -> EMA branch, gamma_bar <- 0.7*3 + 0.3*2 = 2.7, gamma <- ceil(2.7)
    gamma, gamma_bar = gammatune_update(gamma, gamma_bar, 2, eta=eta, delta=delta, gmin=gmin, gmax=gmax)
    assert gamma == 3, gamma
    assert abs(gamma_bar - 2.7) < 1e-9, gamma_bar

    # round 3: A = 3 == gamma(3) -> expand branch again, gamma <- 3 + 2, gamma_bar stays 2.7
    gamma, gamma_bar = gammatune_update(gamma, gamma_bar, 3, eta=eta, delta=delta, gmin=gmin, gmax=gmax)
    assert gamma == 5, gamma
    assert abs(gamma_bar - 2.7) < 1e-9, gamma_bar


def test_expand_branch_does_not_touch_ema_and_clips():
    # A == gamma at the ceiling: A + delta overshoots gamma_max -> clipped, EMA untouched
    g, gb = gammatune_update(10, 4.2, 10, delta=2, gmin=1, gmax=10)
    assert g == 10
    assert abs(gb - 4.2) < 1e-9


def test_ema_branch_clips_low():
    # a run of total rejections drags gamma_bar toward gamma_min and never below it
    g, gb = 3, 1.2
    for _ in range(20):
        g, gb = gammatune_update(g, gb, 0, eta=0.3, gmin=1, gmax=10)
    assert gb >= 1.0 - 1e-9
    assert g == 1


def test_zero_accepts_is_ema_branch_even_when_gamma_min():
    # gamma == 1 and A == 0: A != gamma so this is the EMA branch, not expand
    g, gb = gammatune_update(1, 1.0, 0, eta=0.3, gmin=1, gmax=10)
    assert g == 1
    assert gb >= 1.0 - 1e-9  # clipped up to gmin


# --------------------------------------------------------------------------- #
# FakeModel: no real weights, position-dependent one-hot logits
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """logits[0, p, :] is a big logit at argmax_at(p) for every position p."""

    def __init__(self, argmax_at, vocab=8):
        self.argmax_at = argmax_at
        self.vocab = vocab
        self.generation_config = type("g", (), {"eos_token_id": None})()

    def parameters(self):
        return iter([torch.zeros(1)])  # gammatune_generate reads .device off this

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
    # stand-in for encode_prompt: 2-token context, ignores prompt text
    return torch.zeros((1, 2), dtype=torch.long)


def test_gammatune_generate_trace_tracks_accept_pattern(monkeypatch):
    import gammatune

    monkeypatch.setattr(gammatune, "encode_prompt", _seq_ctx)

    # draft and target agree everywhere -> every draft accepted every round ->
    # A == gamma each round -> gamma ratchets up: 3, 5, 7, 9, 10(clip), 10 ...
    draft = FakeModel(lambda p: 1)
    target = FakeModel(lambda p: 1)

    res = gammatune.gammatune_generate(
        "x", draft, target, _Tok(),
        config=GammaTuneConfig(), max_new_tokens=40, temperature=0.0, seed=0,
    )
    assert res.gamma_trace[0] == 3
    assert res.gamma_trace[1] == 5
    assert res.gamma_trace[2] == 7
    assert max(res.gamma_trace) <= 10
    # all-accept every round: mean_accept_length equals the mean window size,
    # mean_emitted is that + 1
    assert res.mean_emitted_per_round == res.mean_accept_length + 1.0
    assert res.final_state[0] >= 3


def test_carry_state_seeds_controller(monkeypatch):
    import gammatune

    monkeypatch.setattr(gammatune, "encode_prompt", _seq_ctx)
    draft = FakeModel(lambda p: 1)
    target = FakeModel(lambda p: 1)

    # start already warmed up near the ceiling
    res = gammatune.gammatune_generate(
        "x", draft, target, _Tok(),
        config=GammaTuneConfig(), max_new_tokens=12, temperature=0.0, seed=0,
        carry_state=(9, 8.5),
    )
    assert res.gamma_trace[0] == 9  # used the carried gamma, not config.gamma_init (3)


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
