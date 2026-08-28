"""
P6.1 -- batched, KV-cache-correct speculative decoding (EQSPEC-style resync).

`notes/deployment-depth-plan_2026-08-28.md` sec 7 C3: every naive batched
speculative-decoding route (Masking / Rollback / Dynamic Padding) breaks output
equivalence, because within one batch each sequence accepts a *different* number
of drafts, so position ids / attention mask / KV-cache length go ragged and drift
across rounds. The paper's fix, EQSPEC, re-establishes a rectangular committed
invariant after every verify round (correct, but the realignment costs ~40% of
the compute at batch size 8).

This module takes the EQSPEC invariant to its limit: **per-sequence KV caches**.
Each active sequence carries its own draft/target `DynamicCache` and its own
committed-token list, and one "batched round" advances every sequence by exactly
one `speculative_step_kv` (src/spec_kv.py) against *its own* caches. The
rectangular invariant then holds trivially -- there is no shared ragged tensor to
resync -- and every sequence is bit-for-bit the single-sequence KV path
(`tests/test_spec_kv_batch.py` pins `batch == 1-at-a-time`, temperature 0).

What that costs, and what we measure instead of paying it:
  * No kernel-level batching of the ragged verify forward, so wall-clock
    throughput is ~flat in the number of concurrent sequences on this path
    (`src/verify_serving_loop.py`). That flatness *is* the finding: a real
    batched speedup needs the shared ragged tensor back, i.e. it needs to pay
    EQSPEC's realignment tax.
  * `ragged_realignment_overhead()` quantifies that tax analytically for the
    counterfactual rectangular padded batch: given the per-sequence work sizes
    of a round (pending tokens + gamma), a padded batch forwards
    `n_active * max(work)` positions to do `sum(work)` useful ones, so the
    wasted fraction is `1 - sum(work) / (n_active * max(work))`. Reported per
    concurrency level in `results/p6_1_serving_throughput.json`.

Hermetic parity uses the deterministic FakeModel from src/spec_oracles.py; the
same code runs the real Qwen pair in the verify script.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from rejection_sampling import Injection, dist_from_logits, encode_prompt, _sample
from spec_kv import _cache_position, _new_cache, speculative_step_kv


# --------------------------------------------------------------------------- #
# per-sequence state
# --------------------------------------------------------------------------- #
@dataclass
class SeqState:
    """One in-flight generation. Holds its own KV caches so a round never needs
    to reconcile ragged batch tensors."""

    req_id: str
    committed: List[int]                 # prompt + emitted, full history
    prompt_len: int
    max_new_tokens: int
    generator: torch.Generator
    draft_cache: object
    target_cache: object
    draft_synced: int = 0
    target_synced: int = 0
    token_ids: List[int] = field(default_factory=list)
    accept_lengths: List[int] = field(default_factory=list)
    accepted_total: int = 0
    evaluated_total: int = 0
    n_rounds: int = 0
    done: bool = False
    eos_hit: bool = False
    draft_stale: bool = False            # set by a degraded round; cleared by the next spec round
    submit_round: int = 0
    admit_round: int = -1
    finish_round: int = -1

    @property
    def alpha(self) -> float:
        return self.accepted_total / self.evaluated_total if self.evaluated_total else 0.0


@dataclass
class RoundTelemetry:
    n_active: int
    mode: str                            # "spec" | "degraded"
    round_gamma: int
    emitted_per_seq: List[int]
    accepted_per_seq: List[int]
    evaluated_per_seq: List[int]
    work_units: List[int]                # verify-forward positions per seq (pending + gamma)
    realignment_overhead: float          # wasted fraction of a padded rectangular batch
    finished_this_round: List[str]
    wall_s: float


# --------------------------------------------------------------------------- #
# realignment-tax model (EQSPEC's ~40% at BS=8, computed not paid)
# --------------------------------------------------------------------------- #
def ragged_realignment_overhead(work_units: List[int]) -> float:
    """Fraction of a rectangular padded-batch forward that would be padding.

    A padded batch of `n` sequences whose true per-sequence forward lengths are
    `work_units` still has to run `n * max(work_units)` positions to cover the
    `sum(work_units)` real ones. Returns `1 - sum / (n * max)` in [0, 1);
    0 when every sequence does equal work, ->1 as the accept lengths spread out.
    """
    if not work_units:
        return 0.0
    n = len(work_units)
    mx = max(work_units)
    if mx == 0:
        return 0.0
    return 1.0 - sum(work_units) / (n * mx)


# --------------------------------------------------------------------------- #
# one round over a set of sequences
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _degraded_target_step(seq: SeqState, target_model, temperature: float,
                          device, dtype) -> List[int]:
    """One plain KV-cached target forward for `seq` (circuit-breaker degraded
    mode). Leaves the draft cache untouched; `speculative_step_kv` re-syncs it
    from `draft_synced` on the next speculative round."""
    pending = seq.committed[seq.target_synced:]
    feed = torch.tensor([pending], device=device, dtype=dtype)
    pos = _cache_position(seq.target_synced, len(pending), device)
    logits = target_model(input_ids=feed, past_key_values=seq.target_cache,
                          use_cache=True, cache_position=pos).logits
    seq.target_synced += len(pending)
    row = dist_from_logits(logits[0, -1, :], temperature)
    tok = _sample(row, seq.generator)
    return [tok]


@torch.no_grad()
def run_round(seqs: List[SeqState], draft_model, target_model, *, gamma: int,
              temperature: float, eos_ids, device, dtype, mode: str = "spec",
              injection: Optional[Injection] = None) -> RoundTelemetry:
    """Advance every not-done sequence in `seqs` by one round. `mode="degraded"`
    runs a single target step per sequence instead of a speculative round."""
    active = [s for s in seqs if not s.done]
    emitted_per_seq: List[int] = []
    accepted_per_seq: List[int] = []
    evaluated_per_seq: List[int] = []
    work_units: List[int] = []
    finished: List[str] = []
    t0 = time.perf_counter()

    for s in active:
        budget = s.max_new_tokens - len(s.token_ids)
        g = min(gamma, budget)
        pending_target = len(s.committed) - s.target_synced
        work_units.append(pending_target + g)

        if mode == "degraded":
            emitted = _degraded_target_step(s, target_model, temperature, device, dtype)
            n_acc = n_eval = 0
            s.draft_stale = True
        else:
            step = speculative_step_kv(
                s.committed, draft_model, target_model, s.draft_cache, s.target_cache,
                s.draft_synced, s.target_synced, g, device=device, dtype=dtype,
                temperature=temperature, generator=s.generator, injection=injection,
            )
            s.draft_synced = step.draft_synced
            s.target_synced = step.target_synced
            s.draft_stale = False
            emitted = list(step.result.new_token_ids)
            n_acc = step.result.n_accepted
            n_eval = step.result.n_evaluated

        # 坑13: EOS can land mid-block; truncate at (and keep) the first one.
        hit = False
        for k, tid in enumerate(emitted):
            if tid in eos_ids:
                emitted = emitted[: k + 1]
                hit = True
                break

        s.committed.extend(emitted)
        s.token_ids.extend(emitted)
        s.accept_lengths.append(n_acc)
        s.accepted_total += n_acc
        s.evaluated_total += n_eval
        s.n_rounds += 1
        emitted_per_seq.append(len(emitted))
        accepted_per_seq.append(n_acc)
        evaluated_per_seq.append(n_eval)

        if hit or len(s.token_ids) >= s.max_new_tokens:
            s.done = True
            s.eos_hit = hit
            finished.append(s.req_id)

    wall = time.perf_counter() - t0
    return RoundTelemetry(
        n_active=len(active),
        mode=mode,
        round_gamma=gamma,
        emitted_per_seq=emitted_per_seq,
        accepted_per_seq=accepted_per_seq,
        evaluated_per_seq=evaluated_per_seq,
        work_units=work_units,
        realignment_overhead=ragged_realignment_overhead(work_units),
        finished_this_round=finished,
        wall_s=wall,
    )


# --------------------------------------------------------------------------- #
# helper: build a fresh SeqState
# --------------------------------------------------------------------------- #
def make_seq(req_id: str, prompt: str, tokenizer, *, device, max_new_tokens: int,
             seed: int, apply_chat_template: bool = True,
             make_cache: Callable[[], object] = _new_cache) -> SeqState:
    ctx = encode_prompt(tokenizer, prompt, device, apply_chat_template)
    committed = ctx[0].tolist()
    gen = torch.Generator()
    gen.manual_seed(seed)
    return SeqState(
        req_id=req_id,
        committed=committed,
        prompt_len=len(committed),
        max_new_tokens=max_new_tokens,
        generator=gen,
        draft_cache=make_cache(),
        target_cache=make_cache(),
    )


def assert_rectangular_invariant(seqs: List[SeqState]) -> None:
    """EQSPEC's post-round invariant, per sequence: each KV cache holds exactly
    the committed prefix minus its one un-fed trailing token."""
    for s in seqs:
        if s.done:
            continue
        want = len(s.committed) - 1
        dl = s.draft_cache.get_seq_length()
        tl = s.target_cache.get_seq_length()
        assert tl == want, f"{s.req_id}: target cache {tl} != committed-1 ({want})"
        if s.draft_stale:
            assert dl <= want, f"{s.req_id}: stale draft cache {dl} > committed-1 ({want})"
        else:
            assert dl == want, f"{s.req_id}: draft cache {dl} != committed-1 ({want})"
