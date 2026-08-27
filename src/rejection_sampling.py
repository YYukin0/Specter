"""
P1.1 -- Probabilistic rejection-sampling speculative decoding (Leviathan et al. 2023).

Implements the standard algorithm from notes/project_plan_v9.md appendix A.1
*verbatim* (do not re-derive from memory):

    for i in 1..gamma:
        draft_token[i] ~ p_DM(. | context, draft_token[1..i-1])
    target_probs = TargetModel.forward(context, draft_token[1..gamma])   # one forward
    n_accepted = 0
    for i in 1..gamma:
        r ~ Uniform(0, 1)
        if r < min(1, p_TM(x_i) / p_DM(x_i)):  accept; n_accepted += 1
        else:                                  break
    if n_accepted < gamma:
        x_new ~ norm(max(0, p_TM - p_DM))          # resample from the adjusted distribution
    else:
        x_new ~ p_TM(. | context, draft_token[1..gamma])   # bonus token, from the TARGET model

坑2 (the single most common project-wide bug, notes/project_plan_v9.md sec 9.2):
the bonus token MUST come from the *target* model's distribution, never the draft
model's. Sampling it from the draft model does not crash and does not raise -- it
silently biases the output distribution toward the draft model's preferences and
breaks the correctness guarantee (the emitted distribution is no longer equal to
running the target model alone). The `Injection` hooks below reproduce that bug
(and a forced-accept bug) on purpose so P1.2's verifier can be *shown* to catch
them (notes/project_plan_v9.md sec 9.6 risk 3).

Design notes:
- temperature == 0 collapses each distribution to a one-hot vector at its argmax,
  so greedy decoding is a genuine special case of this same code path, not a
  separate branch. That is what lets P1.2 compare greedy speculative output
  against a plain `target_model.generate(do_sample=False)` run token for token.
- All randomness flows through a single CPU `torch.Generator`, so a given `seed`
  reproduces a run bit for bit regardless of MPS/CPU placement.
- The draft model is re-run over the whole prefix each of the gamma steps (no KV
  cache). That is O(gamma) small-model forwards per round; it keeps the code
  obviously correct. KV-cache reuse is a throughput concern deferred to P4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


# --------------------------------------------------------------------------- #
# Fault injection (P1.2 reverse check only -- never set on the real path)
# --------------------------------------------------------------------------- #
@dataclass
class Injection:
    """Deliberate correctness-breaking bugs for the P1.2 fault-injection tests.

    Every field defaults to "off". `Injection()` is a no-op and is what the
    production decode path always uses.
    """

    # 坑2: draw the bonus token from the draft model's distribution instead of
    # the target model's. Only bites on rounds where all gamma drafts are accepted.
    bonus_from_draft: bool = False
    # Force the draft token at this within-round index to be accepted even when
    # the acceptance test says reject. Only bites when that draft would otherwise
    # have been rejected.
    force_accept_index: Optional[int] = None


# --------------------------------------------------------------------------- #
# Pure helpers -- unit-testable with tiny hand-built probability vectors
# --------------------------------------------------------------------------- #
def dist_from_logits(logits_row: torch.Tensor, temperature: float) -> torch.Tensor:
    """Next-token distribution as a 1-D float32 tensor.

    temperature == 0 -> one-hot at the argmax (greedy).
    """
    logits_row = logits_row.float()
    if temperature == 0.0:
        probs = torch.zeros_like(logits_row)
        probs[int(torch.argmax(logits_row))] = 1.0
        return probs
    return torch.softmax(logits_row / temperature, dim=-1)


def adjusted_distribution(p_dm: torch.Tensor, p_tm: torch.Tensor) -> torch.Tensor:
    """p'_TM = norm(max(0, p_TM - p_DM)) -- the distribution to resample from on rejection.

    If the residual mass is ~0 (the draft already covers the target everywhere,
    which happens with one-hot greedy vectors when draft == target argmax), fall
    back to p_TM so the return value is always a valid distribution.
    """
    residual = torch.clamp(p_tm - p_dm, min=0.0)
    total = residual.sum()
    if total < 1e-12:
        return p_tm.clone()
    return residual / total


def acceptance_probability(p_dm_x: float, p_tm_x: float) -> float:
    """min(1, p_TM(x) / p_DM(x)). p_DM(x) == 0 means the draft could not have drawn x."""
    p_dm_x = float(p_dm_x)
    p_tm_x = float(p_tm_x)
    if p_dm_x <= 0.0:
        return 1.0
    return min(1.0, p_tm_x / p_dm_x)


def _sample(dist: torch.Tensor, generator: torch.Generator) -> int:
    """Multinomial draw on CPU for device-independent reproducibility."""
    return int(torch.multinomial(dist.detach().cpu(), num_samples=1, generator=generator).item())


def _uniform(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator).item())


def encode_prompt(tokenizer, prompt: str, device, apply_chat_template: bool) -> torch.Tensor:
    """Return a [1, L] input-id tensor. `apply_chat_template` with return_tensors="pt"
    yields a bare tensor on some transformers versions and a BatchEncoding on others;
    handle both."""
    if apply_chat_template:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids.to(device)


# --------------------------------------------------------------------------- #
# Per-round result
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    new_token_ids: List[int]
    n_accepted: int                 # drafts accepted this round (0..gamma)
    gamma: int
    n_evaluated: int                # drafts actually given an acceptance test
    from_bonus: bool                # True: last token is a target bonus token
    proposals: List[dict] = field(default_factory=list)  # populated only when record=True


@torch.no_grad()
def speculative_step(
    context: torch.Tensor,
    draft_model,
    target_model,
    gamma: int,
    *,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
    record: bool = False,
    injection: Optional[Injection] = None,
) -> StepResult:
    """One speculative round: propose gamma drafts, verify in a single target forward,
    emit the accepted prefix plus one resampled-or-bonus token.

    `context` is a [1, L] token-id tensor on the model device. Returns a StepResult;
    the caller is responsible for appending `new_token_ids` to the context.
    """
    if generator is None:
        generator = torch.Generator()
    if injection is None:
        injection = Injection()

    device = context.device
    ctx_len = context.shape[1]

    # 1. draft model proposes gamma tokens autoregressively -------------------
    draft_tokens: List[int] = []
    p_dm_rows: List[torch.Tensor] = []
    cur = context
    for _ in range(gamma):
        row = dist_from_logits(draft_model(cur).logits[0, -1, :], temperature)
        tok = _sample(row, generator)
        p_dm_rows.append(row)
        draft_tokens.append(tok)
        cur = torch.cat([cur, torch.tensor([[tok]], device=device, dtype=context.dtype)], dim=1)

    # 2. single target forward over context + all gamma drafts ---------------
    candidate = torch.cat(
        [context, torch.tensor([draft_tokens], device=device, dtype=context.dtype)], dim=1
    )
    target_logits = target_model(candidate).logits[0]  # [L+gamma, V]
    # row j (j = 0..gamma) predicts the (j+1)-th new token; row gamma is the bonus slot
    p_tm_rows = [dist_from_logits(target_logits[ctx_len - 1 + j, :], temperature) for j in range(gamma + 1)]

    # 3. rejection-sampling acceptance loop ----------------------------------
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
                    # per-position theoretical acceptance prob = sum_x min(p_DM, p_TM)
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

    # 4. emit accepted prefix + one token ----------------------------------
    if n_accepted == gamma:
        # draft distribution at the bonus position: needed by the 坑2 injection, and
        # also logged when record=True so P1.3 can test the bonus token's provenance
        # (does it follow p_TM or p_DM?).
        draft_bonus_row = None
        if injection.bonus_from_draft or record:
            draft_bonus_row = dist_from_logits(draft_model(cur).logits[0, -1, :], temperature)
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

    return StepResult(
        new_token_ids=new_tokens,
        n_accepted=n_accepted,
        gamma=gamma,
        n_evaluated=n_evaluated,
        from_bonus=from_bonus,
        proposals=proposals,
    )


# --------------------------------------------------------------------------- #
# Full generation loop
# --------------------------------------------------------------------------- #
@dataclass
class GenResult:
    text: str
    token_ids: List[int]
    n_rounds: int
    accept_lengths: List[int]     # n_accepted per round (0..gamma)
    emitted_per_round: List[int]  # tokens emitted per round (n_accepted + 1)
    accepted_total: int           # accepted drafts (alpha numerator)
    evaluated_total: int          # drafts given an acceptance test (alpha denominator)
    alpha: float
    elapsed_s: float
    proposals: List[dict] = field(default_factory=list)  # only when record=True


def collect_eos_ids(tokenizer, model) -> set:
    ids = set()
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    for e in (tok_eos if isinstance(tok_eos, list) else [tok_eos]):
        if e is not None:
            ids.add(int(e))
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for e in (gen_eos if isinstance(gen_eos, list) else [gen_eos]):
        if e is not None:
            ids.add(int(e))
    return ids


@torch.no_grad()
def speculative_generate(
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
) -> GenResult:
    """Repeatedly call `speculative_step` until `max_new_tokens` or an EOS token.

    alpha is reported as accepted_total / evaluated_total, where evaluated counts
    every draft that was actually given an acceptance test (accepted ones plus the
    first rejected one per round) -- the same convention as src/gate_p1_0.py.
    """
    if injection is None:
        injection = Injection()
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device

    context = encode_prompt(tokenizer, prompt, device, apply_chat_template)

    eos_ids = collect_eos_ids(tokenizer, target_model)

    token_ids: List[int] = []
    accept_lengths: List[int] = []
    emitted_per_round: List[int] = []
    proposals: List[dict] = []
    accepted_total = 0
    evaluated_total = 0

    t0 = time.perf_counter()
    while len(token_ids) < max_new_tokens:
        g = min(gamma, max_new_tokens - len(token_ids))
        step = speculative_step(
            context,
            draft_model,
            target_model,
            g,
            temperature=temperature,
            generator=generator,
            record=record,
            injection=injection,
        )
        # A round emits several tokens at once; an EOS can land in the middle of
        # that chunk. Truncate at (and keep) the first EOS so speculative decoding
        # stops at exactly the same place plain autoregressive decoding would --
        # without this, greedy speculative output runs past EOS and diverges from
        # the target-only reference (P1.2).
        emitted = step.new_token_ids
        hit_eos = False
        for k, tid in enumerate(emitted):
            if tid in eos_ids:
                emitted = emitted[: k + 1]
                hit_eos = True
                break

        token_ids.extend(emitted)
        accept_lengths.append(step.n_accepted)
        emitted_per_round.append(len(emitted))
        accepted_total += step.n_accepted
        evaluated_total += step.n_evaluated
        if record:
            proposals.extend(step.proposals)
        context = torch.cat(
            [context, torch.tensor([emitted], device=device, dtype=context.dtype)], dim=1
        )
        if hit_eos:
            break
    elapsed = time.perf_counter() - t0

    return GenResult(
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
    )


@torch.no_grad()
def target_only_generate(
    prompt: str,
    target_model,
    tokenizer,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
    apply_chat_template: bool = True,
) -> GenResult:
    """Plain autoregressive target-model decoding, same sampling convention as
    `speculative_generate` (single shared CPU generator, temperature == 0 -> greedy).
    Used as the correctness/parity reference in P1.2 and P1.3.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device

    context = encode_prompt(tokenizer, prompt, device, apply_chat_template)

    eos_ids = collect_eos_ids(tokenizer, target_model)
    token_ids: List[int] = []
    t0 = time.perf_counter()
    while len(token_ids) < max_new_tokens:
        row = dist_from_logits(target_model(context).logits[0, -1, :], temperature)
        tok = _sample(row, generator)
        token_ids.append(tok)
        context = torch.cat(
            [context, torch.tensor([[tok]], device=device, dtype=context.dtype)], dim=1
        )
        if tok in eos_ids:
            break
    elapsed = time.perf_counter() - t0

    return GenResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        n_rounds=len(token_ids),
        accept_lengths=[],
        emitted_per_round=[],
        accepted_total=0,
        evaluated_total=0,
        alpha=0.0,
        elapsed_s=elapsed,
    )
