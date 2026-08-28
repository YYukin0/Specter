"""P6.1 -- batched KV speculative decoding: output-equivalence + resync invariant.

The whole point of the per-sequence-cache design (src/spec_kv_batch.py) is that
a batched round cannot break output equivalence, because there is no shared
ragged tensor to drift. These tests pin:

  (1) a round driven over N sequences emits exactly what N independent
      single-sequence `speculative_generate_kv` runs would, token for token,
      at matched seeds (greedy AND sampling);
  (2) EQSPEC's rectangular invariant holds per sequence after every round --
      each KV cache is exactly `len(committed) - 1`;
  (3) `ragged_realignment_overhead` matches its closed form and is 0 iff every
      sequence did equal work.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spec_kv import speculative_generate_kv  # noqa: E402
from spec_kv_batch import (  # noqa: E402
    assert_rectangular_invariant,
    make_seq,
    ragged_realignment_overhead,
    run_round,
)
from spec_oracles import LengthOnlyCache, make_fake_pair  # noqa: E402
from rejection_sampling import collect_eos_ids  # noqa: E402
import torch  # noqa: E402

PROMPTS = ["hello world", "speculative decode", "a longer prompt here", "kv",
           "the quick brown fox", "one two three four"]


def _drive(seqs, draft, target, tok, *, gamma, temperature, rounds=10**6):
    eos = collect_eos_ids(tok, target)
    dev = torch.device("cpu")
    n = 0
    while any(not s.done for s in seqs) and n < rounds:
        run_round(seqs, draft, target, gamma=gamma, temperature=temperature,
                  eos_ids=eos, device=dev, dtype=torch.float32, mode="spec")
        assert_rectangular_invariant(seqs)
        n += 1
    return seqs


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_batched_round_matches_single_sequence(temperature):
    draft, target, tok = make_fake_pair()
    gamma, mnt = 3, 32

    seqs = [make_seq(f"s{i}", p, tok, device=torch.device("cpu"), max_new_tokens=mnt,
                     seed=10 + i, make_cache=LengthOnlyCache)
            for i, p in enumerate(PROMPTS)]
    _drive(seqs, draft, target, tok, gamma=gamma, temperature=temperature)

    for i, p in enumerate(PROMPTS):
        ref = speculative_generate_kv(p, draft, target, tok, make_cache=LengthOnlyCache,
                                      gamma=gamma, max_new_tokens=mnt,
                                      temperature=temperature, seed=10 + i)
        assert seqs[i].token_ids == ref.token_ids, (
            f"{p!r}: batched round diverged from single-sequence at temp={temperature}"
        )
        assert seqs[i].accept_lengths == ref.accept_lengths


def test_rectangular_invariant_holds_every_round():
    draft, target, tok = make_fake_pair(phase_target=1.2)  # lots of rejections
    seqs = [make_seq(f"s{i}", p, tok, device=torch.device("cpu"), max_new_tokens=40,
                     seed=i, make_cache=LengthOnlyCache)
            for i, p in enumerate(PROMPTS)]
    eos = collect_eos_ids(tok, target)
    while any(not s.done for s in seqs):
        run_round(seqs, draft, target, gamma=5, temperature=1.0, eos_ids=eos,
                  device=torch.device("cpu"), dtype=torch.float32, mode="spec")
        assert_rectangular_invariant(seqs)  # raises on any drift


def test_sequences_finish_at_different_rounds_and_batch_drains():
    draft, target, tok = make_fake_pair()
    budgets = [8, 16, 24, 40]
    seqs = [make_seq(f"s{i}", PROMPTS[i], tok, device=torch.device("cpu"),
                     max_new_tokens=b, seed=i, make_cache=LengthOnlyCache)
            for i, b in enumerate(budgets)]
    eos = collect_eos_ids(tok, target)
    finish_round = {}
    r = 0
    while any(not s.done for s in seqs):
        tele = run_round(seqs, draft, target, gamma=3, temperature=1.0, eos_ids=eos,
                         device=torch.device("cpu"), dtype=torch.float32, mode="spec")
        for rid in tele.finished_this_round:
            finish_round[rid] = r
        r += 1
    assert all(s.done for s in seqs)
    # shorter-budget sequences retire strictly earlier
    assert finish_round["s0"] <= finish_round["s1"] <= finish_round["s3"]
    assert finish_round["s0"] < finish_round["s3"]
    # a round emits up to gamma+1 tokens, so the budget overshoot is bounded by gamma
    assert all(b <= len(s.token_ids) <= b + 3 for s, b in zip(seqs, budgets))


def test_realignment_overhead_closed_form():
    assert ragged_realignment_overhead([]) == 0.0
    assert ragged_realignment_overhead([4, 4, 4]) == 0.0
    # 3 seqs, work [1, 2, 6] -> padded batch runs 3*6=18 to do 9 useful -> 0.5 waste
    assert ragged_realignment_overhead([1, 2, 6]) == pytest.approx(0.5)
    # spread grows the tax
    assert ragged_realignment_overhead([1, 1, 10]) > ragged_realignment_overhead([3, 4, 5])


def test_degraded_round_is_plain_target_decoding():
    draft, target, tok = make_fake_pair()
    # one sequence, degraded server round == one target_only_generate_kv token
    from spec_kv import target_only_generate_kv
    seq = make_seq("s0", "a longer prompt here", tok, device=torch.device("cpu"),
                   max_new_tokens=12, seed=3, make_cache=LengthOnlyCache)
    eos = collect_eos_ids(tok, target)
    while not seq.done:
        run_round([seq], draft, target, gamma=4, temperature=0.0, eos_ids=eos,
                  device=torch.device("cpu"), dtype=torch.float32, mode="degraded")
    ref = target_only_generate_kv("a longer prompt here", target, tok,
                                  make_cache=LengthOnlyCache, max_new_tokens=12,
                                  temperature=0.0, seed=3)
    assert seq.token_ids == ref.token_ids


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
