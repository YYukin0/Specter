"""
P1.1 unit tests -- run with `pytest tests/` or `python tests/test_rejection_sampling.py`.

The core test (`test_appendix_a1_numerical_example`) reproduces the hand-worked
numeric example in notes/project_plan_v9.md appendix A.1 and asserts the exact
numbers (acceptance prob 0.571; adjusted distribution {A:0, B:0.33, C:0.67} with
A's mass driven to exactly zero). This is the "known-correct" reference point that
P1.2's fault-injection tests build on -- see the plan, "bonus token 测试通过".

The FakeModel tests exercise `speculative_step` end to end without loading any real
weights and pin down 坑2: the bonus token must come from the target model, and
flipping it to the draft model (via `Injection`) must visibly change the output.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rejection_sampling import (  # noqa: E402
    Injection,
    acceptance_probability,
    adjusted_distribution,
    speculative_step,
)

A, B, C = 0, 1, 2  # 3-token toy vocab from appendix A.1


# --------------------------------------------------------------------------- #
# Appendix A.1 hand-worked example
# --------------------------------------------------------------------------- #
def test_appendix_a1_numerical_example():
    p_dm = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    p_tm = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64)

    # draft sampled A; acceptance prob = min(1, p_TM(A) / p_DM(A)) = min(1, 0.4/0.7)
    a = acceptance_probability(p_dm[A].item(), p_tm[A].item())
    assert abs(a - 0.5714285714) < 1e-9, a
    assert round(a, 3) == 0.571  # the value quoted in appendix A.1

    # r = 0.8 > 0.571 -> reject; r = 0.5 < 0.571 -> accept
    assert not (0.8 < a)
    assert 0.5 < a

    # adjusted distribution norm(max(0, p_TM - p_DM)) = {A:0, B:1/3, C:2/3}
    adj = adjusted_distribution(p_dm, p_tm)
    assert adj[A].item() == 0.0, "坑2: A's mass must be driven to exactly zero on rejection"
    assert abs(adj[B].item() - 1.0 / 3.0) < 1e-6, adj[B].item()
    assert abs(adj[C].item() - 2.0 / 3.0) < 1e-6, adj[C].item()
    assert abs(adj.sum().item() - 1.0) < 1e-6


def test_acceptance_probability_edges():
    assert acceptance_probability(0.2, 0.5) == 1.0        # p_TM >= p_DM -> always accept
    assert acceptance_probability(0.0, 0.3) == 1.0        # draft could not have drawn x
    assert abs(acceptance_probability(0.8, 0.2) - 0.25) < 1e-9


def test_adjusted_distribution_fallback_to_target():
    # residual all <= 0 (here draft == target, as with one-hot greedy vectors that
    # match) -> fall back to p_TM so the return value is still a valid distribution
    p_dm = torch.tensor([0.5, 0.3, 0.2])
    p_tm = torch.tensor([0.5, 0.3, 0.2])
    adj = adjusted_distribution(p_dm, p_tm)
    assert torch.allclose(adj, p_tm)


# --------------------------------------------------------------------------- #
# FakeModel: no real weights, position-dependent logits
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns logits[0, p, :] = one-hot-ish (big logit) at `argmax_at(p)` for
    every position p in the input. `argmax_at` maps absolute sequence position to
    the token id this model would greedily predict as the *next* token."""

    def __init__(self, argmax_at, vocab=3):
        self.argmax_at = argmax_at
        self.vocab = vocab
        self.generation_config = type("g", (), {"eos_token_id": None})()

    def __call__(self, input_ids):
        seqlen = input_ids.shape[1]
        logits = torch.full((1, seqlen, self.vocab), -10.0)
        for p in range(seqlen):
            logits[0, p, self.argmax_at(p)] = 10.0
        return _Out(logits)


def _ctx(length=2):
    return torch.zeros((1, length), dtype=torch.long)


def test_bonus_token_comes_from_target_not_draft():
    # ctx_len = 2, gamma = 2. Draft proposes token 1 at both new positions.
    # Target agrees at positions 1,2 (rows read at abs pos 1 and 2) so both drafts
    # accept; bonus slot (abs pos 3) -> target says token C(2), draft says token A(0).
    draft = FakeModel(lambda p: B)                       # always proposes B
    target = FakeModel(lambda p: B if p in (1, 2) else C)  # agrees on drafts, bonus -> C

    good = speculative_step(_ctx(), draft, target, gamma=2, temperature=0.0)
    assert good.n_accepted == 2
    assert good.from_bonus is True
    assert good.new_token_ids == [B, B, C], good.new_token_ids  # bonus from TARGET

    # 坑2 injected: bonus drawn from the draft model instead. Draft at abs pos 3 -> A.
    draft_bonus = FakeModel(lambda p: B if p < 3 else A)
    bad = speculative_step(
        _ctx(), draft_bonus, target, gamma=2, temperature=0.0,
        injection=Injection(bonus_from_draft=True),
    )
    assert bad.new_token_ids == [B, B, A], bad.new_token_ids  # bonus leaked from DRAFT
    assert good.new_token_ids != bad.new_token_ids


def test_force_accept_injection_changes_output():
    # Draft proposes B,B. Target would reject at index 0 (wants A there).
    draft = FakeModel(lambda p: B)
    target = FakeModel(lambda p: A if p == 1 else (B if p == 2 else C))

    normal = speculative_step(_ctx(), draft, target, gamma=2, temperature=0.0)
    assert normal.n_accepted == 0
    assert normal.new_token_ids == [A], normal.new_token_ids  # resampled -> target's A

    forced = speculative_step(
        _ctx(), draft, target, gamma=2, temperature=0.0,
        injection=Injection(force_accept_index=0),
    )
    # index 0 forced in; index 1 (B) matches target -> all accepted -> bonus C
    assert forced.new_token_ids == [B, B, C], forced.new_token_ids
    assert forced.new_token_ids != normal.new_token_ids


def test_greedy_all_accept_no_bonus_bug():
    # draft == target everywhere -> every round fully accepts, bonus from target
    draft = FakeModel(lambda p: B)
    target = FakeModel(lambda p: B)
    res = speculative_step(_ctx(), draft, target, gamma=3, temperature=0.0)
    assert res.n_accepted == 3
    assert res.new_token_ids == [B, B, B, B]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
