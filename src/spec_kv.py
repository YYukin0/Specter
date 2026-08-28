"""
P6.0 -- KV-cache-correct single-sequence speculative decoding.

`speculative_generate_kv` reproduces `speculative_generate`
(src/rejection_sampling.py) token for token -- a KV cache changes throughput
only, never any observable output -- while keeping a `DynamicCache` for the draft
and target models instead of re-running each model over the whole prefix on every
step.

The whole difficulty is the partial-acceptance rollback. One speculative round:
  * the draft model is stepped `gamma` times (one new token fed per step) -> the
    draft cache grows by `gamma - 1` entries past the round-start prefix (the
    gamma-th draft token is sampled but never fed back to the draft model,
    exactly as in the no-cache code).
  * the target model does ONE forward over [pending target tokens .. gamma
    drafts] -> the target cache grows to `prefix + gamma`.
  * `k` drafts are accepted (0 <= k <= gamma). The committed continuation is
    `draft[:k]` + one token (resampled from the adjusted distribution if
    `k < gamma`, or a target bonus token if `k == gamma`).
  * ROLLBACK: both caches are cropped back to `prefix + k`. This matches the
    Hugging Face assisted-generation reference (generation/utils.py:
    ``number_of_tokens_to_crop = candidate_length - n_matches;
    outputs.past_key_values.crop(-number_of_tokens_to_crop)``), which leaves the
    cache exactly one token behind the committed sequence -- the freshly
    committed last token (resample or bonus) was a model *output*, never fed as
    an *input*, so its KV does not exist yet and it becomes next round's single
    "pending" input token.

Note: the deployment-depth plan sketched the target crop as ``prefix + k + 1``.
The HF reference and a first-principles check both say ``prefix + k`` -- for a
resample the KV cached at position ``prefix + k`` was computed from the *draft*
proposal (a different token) and is stale; for a bonus that position was never an
input at all. Tests pin the ``prefix + k`` behaviour.

Parity contract (= tests/test_spec_kv.py):
  * FakeModel (position-one-hot, deterministic): ``speculative_generate_kv
    (temperature=0)`` == ``speculative_generate(temperature=0)`` ==
    ``target_only_generate``, token for token; and each cache's
    ``get_seq_length()`` == ``len(committed) - 1`` after every round.
  * FakeModel sampling, shared seed: == ``speculative_generate``, token for token
    (identical generator-draw order -- the KV path issues the same
    ``_sample`` / ``_uniform`` calls in the same sequence).

Real MPS fp16 has non-deterministic reductions, so real-model parity is only a
"long common prefix", the same caveat src/spec_batch.py already carries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from rejection_sampling import (
    GenResult,
    Injection,
    StepResult,
    acceptance_probability,
    adjusted_distribution,
    collect_eos_ids,
    dist_from_logits,
    encode_prompt,
    _sample,
    _uniform,
)


# --------------------------------------------------------------------------- #
# Cache helper
# --------------------------------------------------------------------------- #
def _crop_to(cache, target_len: int) -> None:
    """Shrink `cache` so it holds exactly `target_len` positions.

    Wraps `DynamicCache.crop`, whose new (transformers >= 5.16, < 5.18) contract
    is: a *negative* argument removes that many trailing tokens; a positive one
    is the deprecated "absolute length" form and warns. We always drive it with
    the negative form. `target_len >= current` is a no-op (never grows a cache).
    """
    cur = cache.get_seq_length()
    if target_len >= cur:
        return
    cache.crop(-(cur - target_len))


def _cache_position(start: int, n: int, device) -> torch.Tensor:
    return torch.arange(start, start + n, device=device, dtype=torch.long)


# --------------------------------------------------------------------------- #
# One KV-cached speculative round
# --------------------------------------------------------------------------- #
@dataclass
class _KVStep:
    result: StepResult
    draft_synced: int   # draft cache length after rollback
    target_synced: int  # target cache length after rollback


@torch.no_grad()
def speculative_step_kv(
    committed: List[int],
    draft_model,
    target_model,
    draft_cache,
    target_cache,
    draft_synced: int,
    target_synced: int,
    gamma: int,
    *,
    device,
    dtype,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
    record: bool = False,
    injection: Optional[Injection] = None,
) -> _KVStep:
    """One speculative round against live draft/target KV caches.

    `committed` is the full list of already-emitted token ids (prompt included).
    `draft_synced` / `target_synced` are how many leading `committed` tokens each
    cache already holds. Returns the `StepResult` (same shape as
    `speculative_step`) plus the post-rollback cache lengths; the caller appends
    `result.new_token_ids` to `committed`.
    """
    if generator is None:
        generator = torch.Generator()
    if injection is None:
        injection = Injection()

    prefix = len(committed)

    # 1. draft model proposes gamma tokens, one cached forward per token --------
    draft_tokens: List[int] = []
    p_dm_rows: List[torch.Tensor] = []
    pending = committed[draft_synced:]  # >= 1 token (whole prompt on round 0)
    for _ in range(gamma):
        feed = torch.tensor([pending], device=device, dtype=dtype)
        pos = _cache_position(draft_synced, len(pending), device)
        logits = draft_model(
            input_ids=feed,
            past_key_values=draft_cache,
            use_cache=True,
            cache_position=pos,
        ).logits
        draft_synced += len(pending)
        row = dist_from_logits(logits[0, -1, :], temperature)
        tok = _sample(row, generator)
        p_dm_rows.append(row)
        draft_tokens.append(tok)
        pending = [tok]
    # draft_synced == prefix + gamma - 1 here; draft_tokens[-1] not fed back.

    # 2. single target forward over [pending target tokens .. gamma drafts] ----
    tgt_pending = committed[target_synced:]
    t0 = len(tgt_pending)
    tgt_feed = torch.tensor([tgt_pending + draft_tokens], device=device, dtype=dtype)
    tgt_pos = _cache_position(target_synced, t0 + gamma, device)
    tgt_logits = target_model(
        input_ids=tgt_feed,
        past_key_values=target_cache,
        use_cache=True,
        cache_position=tgt_pos,
    ).logits[0]
    target_synced += t0 + gamma  # == prefix + gamma
    # row (t0 - 1 + j) predicts the (j+1)-th new token; j == gamma is the bonus slot
    p_tm_rows = [
        dist_from_logits(tgt_logits[t0 - 1 + j, :], temperature) for j in range(gamma + 1)
    ]

    # 3. rejection-sampling acceptance loop (verbatim from speculative_step) ---
    n_accepted = 0
    resampled_token: Optional[int] = None
    proposals: List[dict] = []
    for i in range(gamma):
        x = draft_tokens[i]
        p_dm_x = float(p_dm_rows[i][x])
        p_tm_x = float(p_tm_rows[i][x])
        a = acceptance_probability(p_dm_x, p_tm_x)
        r = _uniform(generator)
        forced = injection.force_accept_index == i
        accepted = forced or (r < a)

        if record:
            proposals.append(
                {
                    "index": i,
                    "token": x,
                    "p_dm": p_dm_x,
                    "p_tm": p_tm_x,
                    "min_overlap": float(torch.minimum(p_dm_rows[i], p_tm_rows[i]).sum()),
                    "accepted": bool(accepted),
                    "forced": bool(forced),
                }
            )

        if accepted:
            n_accepted += 1
            continue
        resampled_token = _sample(adjusted_distribution(p_dm_rows[i], p_tm_rows[i]), generator)
        break

    n_evaluated = n_accepted + (0 if n_accepted == gamma else 1)

    # 4. emit accepted prefix + one token ------------------------------------
    if n_accepted == gamma:
        # Feed the gamma-th draft token (the one never fed in the draft loop) so
        # the draft cache reaches prefix + gamma and stays exactly one token
        # behind the committed sequence, same as the target cache. This is a pure
        # forward -- no generator draw -- so it cannot perturb output parity. Its
        # logits row is also the draft distribution at the bonus position, needed
        # by the 坑2 injection and by record mode.
        feed = torch.tensor([[draft_tokens[-1]]], device=device, dtype=dtype)
        pos = _cache_position(draft_synced, 1, device)
        b_logits = draft_model(
            input_ids=feed,
            past_key_values=draft_cache,
            use_cache=True,
            cache_position=pos,
        ).logits
        draft_synced += 1
        draft_bonus_row = dist_from_logits(b_logits[0, -1, :], temperature)
        if injection.bonus_from_draft:
            assert draft_bonus_row is not None
            bonus = _sample(draft_bonus_row, generator)  # BUG (坑2): bonus from DRAFT
        else:
            bonus = _sample(p_tm_rows[gamma], generator)
        new_tokens = draft_tokens[:gamma] + [bonus]
        from_bonus = True
        if record:
            assert draft_bonus_row is not None
            proposals.append(
                {
                    "index": "bonus",
                    "token": bonus,
                    "p_dm": float(draft_bonus_row[bonus]),
                    "p_tm": float(p_tm_rows[gamma][bonus]),
                    "min_overlap": float(torch.minimum(draft_bonus_row, p_tm_rows[gamma]).sum()),
                    "accepted": None,
                    "forced": False,
                    "bonus_from_draft": bool(injection.bonus_from_draft),
                }
            )
    else:
        assert resampled_token is not None
        new_tokens = draft_tokens[:n_accepted] + [resampled_token]
        from_bonus = False

    # 5. rollback: both caches back to prefix + k (see module docstring) -------
    keep = prefix + n_accepted
    _crop_to(target_cache, keep)
    _crop_to(draft_cache, keep)
    draft_synced = min(draft_synced, keep)
    target_synced = min(target_synced, keep)

    return _KVStep(
        result=StepResult(
            new_token_ids=new_tokens,
            n_accepted=n_accepted,
            gamma=gamma,
            n_evaluated=n_evaluated,
            from_bonus=from_bonus,
            proposals=proposals,
        ),
        draft_synced=draft_synced,
        target_synced=target_synced,
    )


# --------------------------------------------------------------------------- #
# Full generation loop
# --------------------------------------------------------------------------- #
@dataclass
class KVGenResult(GenResult):
    draft_forwards: int = 0        # draft model call count
    draft_forward_tokens: int = 0  # total tokens fed to the draft model
    target_forwards: int = 0       # target model call count
    peak_cache_len: int = 0


def _new_cache():
    from transformers import DynamicCache

    return DynamicCache()


@torch.no_grad()
def speculative_generate_kv(
    prompt: str,
    draft_model,
    target_model,
    tokenizer,
    *,
    gamma: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
    record: bool = False,
    injection: Optional[Injection] = None,
    apply_chat_template: bool = True,
    make_cache=_new_cache,
) -> KVGenResult:
    """KV-cached speculative decoding. Token-for-token identical to
    `speculative_generate` at the same seed (FakeModel; long-common-prefix on
    real MPS fp16). `make_cache` is injectable so tests can pass a length-only
    fake cache.
    """
    if injection is None:
        injection = Injection()
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device

    context = encode_prompt(tokenizer, prompt, device, apply_chat_template)
    committed: List[int] = context[0].tolist()
    dtype = context.dtype

    eos_ids = collect_eos_ids(tokenizer, target_model)
    draft_cache = make_cache()
    target_cache = make_cache()
    draft_synced = 0
    target_synced = 0

    token_ids: List[int] = []
    accept_lengths: List[int] = []
    emitted_per_round: List[int] = []
    proposals: List[dict] = []
    accepted_total = 0
    evaluated_total = 0
    draft_forwards = 0
    draft_forward_tokens = 0
    target_forwards = 0
    peak_cache_len = 0

    t_start = time.perf_counter()
    while len(token_ids) < max_new_tokens:
        g = min(gamma, max_new_tokens - len(token_ids))
        pending_draft_before = len(committed) - draft_synced

        step = speculative_step_kv(
            committed,
            draft_model,
            target_model,
            draft_cache,
            target_cache,
            draft_synced,
            target_synced,
            g,
            device=device,
            dtype=dtype,
            temperature=temperature,
            generator=generator,
            record=record,
            injection=injection,
        )
        draft_synced = step.draft_synced
        target_synced = step.target_synced

        # draft: 1 forward for the pending chunk + (g-1) single-token forwards,
        # plus 1 more single-token forward on every full-accept round (the
        # gamma-th draft token, fed to keep the draft cache in sync).
        bonus_fwd = 1 if step.result.from_bonus else 0
        draft_forwards += g + bonus_fwd
        draft_forward_tokens += pending_draft_before + (g - 1) + bonus_fwd
        target_forwards += 1
        peak_cache_len = max(peak_cache_len, len(committed) + g)

        # 坑13: an EOS can land mid-chunk; truncate at (and keep) the first one so
        # the KV path stops exactly where plain autoregressive decoding would.
        emitted = step.result.new_token_ids
        hit_eos = False
        for k, tid in enumerate(emitted):
            if tid in eos_ids:
                emitted = emitted[: k + 1]
                hit_eos = True
                break

        token_ids.extend(emitted)
        committed.extend(emitted)
        accept_lengths.append(step.result.n_accepted)
        emitted_per_round.append(len(emitted))
        accepted_total += step.result.n_accepted
        evaluated_total += step.result.n_evaluated
        if record:
            proposals.extend(step.result.proposals)
        if hit_eos:
            break
    elapsed = time.perf_counter() - t_start

    return KVGenResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        n_rounds=len(accept_lengths),
        accept_lengths=accept_lengths,
        emitted_per_round=emitted_per_round,
        accepted_total=accepted_total,
        evaluated_total=evaluated_total,
        alpha=(accepted_total / evaluated_total) if evaluated_total else 0.0,
        elapsed_s=elapsed,
        proposals=proposals,
        draft_forwards=draft_forwards,
        draft_forward_tokens=draft_forward_tokens,
        target_forwards=target_forwards,
        peak_cache_len=peak_cache_len,
    )


@torch.no_grad()
def target_only_generate_kv(
    prompt: str,
    target_model,
    tokenizer,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
    apply_chat_template: bool = True,
    make_cache=_new_cache,
) -> KVGenResult:
    """KV-cached plain autoregressive target decoding -- the *fair* baseline for
    `speculative_generate_kv` (the no-cache `target_only_generate` re-reads the
    whole prefix every token). Same sampling convention: one shared CPU
    generator, temperature == 0 -> greedy.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device

    context = encode_prompt(tokenizer, prompt, device, apply_chat_template)
    committed: List[int] = context[0].tolist()
    dtype = context.dtype
    eos_ids = collect_eos_ids(tokenizer, target_model)

    cache = make_cache()
    synced = 0
    token_ids: List[int] = []
    target_forwards = 0

    t_start = time.perf_counter()
    while len(token_ids) < max_new_tokens:
        pending = committed[synced:]
        feed = torch.tensor([pending], device=device, dtype=dtype)
        pos = _cache_position(synced, len(pending), device)
        logits = target_model(
            input_ids=feed,
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
        ).logits
        synced += len(pending)
        target_forwards += 1
        row = dist_from_logits(logits[0, -1, :], temperature)
        tok = _sample(row, generator)
        token_ids.append(tok)
        committed.append(tok)
        if tok in eos_ids:
            break
    elapsed = time.perf_counter() - t_start

    return KVGenResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        n_rounds=len(token_ids),
        accept_lengths=[],
        emitted_per_round=[],
        accepted_total=0,
        evaluated_total=0,
        alpha=0.0,
        elapsed_s=elapsed,
        draft_forwards=0,
        draft_forward_tokens=0,
        target_forwards=target_forwards,
        peak_cache_len=len(committed),
    )
