"""
M5[A] -- batched speculative decoding throughput curve (local small batch).

Reference: notes/project_plan_v9.md sec 7 M5[A]. The plan's batch sweep {1..64}
is the cloud 7B version; locally (MPS, no KV cache) only {1,2,4,8} is meaningful.

No KV cache (same as src/rejection_sampling.py -- every draft step re-runs the
whole prefix through the draft model, and the target does one forward over
prefix+drafts). Batching just means B sequences share each forward call.

RAGGED handling (documented here and in the results file):
  - Each sequence in the batch accepts a DIFFERENT number of drafts per round
    (accept length is ragged). Emitted tokens per round therefore differ per seq.
  - Sequences finish at different rounds (EOS or max_new_tokens). A finished
    sequence stays in the padded batch tensor (so shapes stay rectangular) but
    is not sampled for and not scored; the round ends when ALL are finished.
  - Left-padding + an explicit attention_mask + explicit position_ids so a
    padded neighbour cannot shift an unpadded sequence's logits (test pins this).
  - Per-sequence remaining-token budget clamps that seq's gamma for the round,
    exactly as the single-sequence loop does (needed for batch_size=1 parity).

The acceptance MATH is not re-implemented: adjusted_distribution /
acceptance_probability / dist_from_logits / _sample are reused verbatim from
rejection_sampling, and the per-round generator-draw ORDER matches
speculative_step (gamma draft draws, then per-draft uniform + optional resample,
then one bonus draw on full accept) so batch_size=1 reproduces
speculative_generate token-for-token at a fixed seed.

Metrics (see BatchGenResult):
  - tokens_per_target_forward : total emitted (all seqs) / number of BATCHED
    target forward calls. Grows with batch (more seqs emit per call) -- this is
    the raw "work per target forward" number.
  - mean_tokens_per_seq_per_round : acceptance efficiency, ~ batch-independent
    without KV cache; a drop as batch grows would be the (weak, local) signal
    that speculation gets relatively worse at high batch.
  - tok_per_s : wall-clock, MPS, NO KV cache -> CAVEAT, indicative only.
  - peak_mem_mb : torch.mps.current_allocated_memory() high-water mark.

Run:  python src/spec_batch.py  (quick self-check)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import torch

from rejection_sampling import (
    acceptance_probability,
    adjusted_distribution,
    collect_eos_ids,
    dist_from_logits,
    encode_prompt,
    _sample,
    _uniform,
)


@dataclass
class SeqState:
    prompt_ids: List[int]
    gen_ids: List[int] = field(default_factory=list)
    finished: bool = False
    accept_lengths: List[int] = field(default_factory=list)
    emitted_per_round: List[int] = field(default_factory=list)
    accepted_total: int = 0
    evaluated_total: int = 0
    rounds: int = 0

    def full(self) -> List[int]:
        return self.prompt_ids + self.gen_ids


@dataclass
class BatchGenResult:
    texts: List[str]
    token_ids: List[List[int]]
    per_seq: List[dict]
    batch_size: int
    n_prompts: int
    n_target_forwards: int          # batched target-forward calls (summed over chunks)
    n_draft_forwards: int
    total_emitted: int
    tokens_per_target_forward: float
    mean_tokens_per_seq_per_round: float
    mean_accept_length: float
    alpha: float
    wall_s: float
    tok_per_s: float                # CAVEAT: MPS, no KV cache
    peak_mem_mb: float
    n_seq_finished_early: int


def _pad_batch(seqs_tokens: List[List[int]], pad_id: int, device, dtype):
    """Left-pad to a common length. Returns (input_ids[B,L], attn[B,L], pos[B,L])."""
    lmax = max(len(t) for t in seqs_tokens)
    B = len(seqs_tokens)
    input_ids = torch.full((B, lmax), pad_id, device=device, dtype=dtype)
    attn = torch.zeros((B, lmax), device=device, dtype=torch.long)
    pos = torch.zeros((B, lmax), device=device, dtype=torch.long)
    for i, t in enumerate(seqs_tokens):
        n = len(t)
        input_ids[i, lmax - n:] = torch.tensor(t, device=device, dtype=dtype)
        attn[i, lmax - n:] = 1
        pos[i, lmax - n:] = torch.arange(n, device=device, dtype=torch.long)
    return input_ids, attn, pos


def _fwd(model, input_ids, attn, pos):
    try:
        return model(input_ids=input_ids, attention_mask=attn, position_ids=pos).logits
    except TypeError:
        # tolerate minimal fake models that only take input_ids
        return model(input_ids).logits


def _peak_mb(cur: float) -> float:
    if torch.backends.mps.is_available():
        return max(cur, torch.mps.current_allocated_memory() / 1e6)
    return cur


@torch.no_grad()
def _run_chunk(states: List[SeqState], draft_model, target_model, *, gamma: int,
               max_new_tokens: int, temperature: float, generator: torch.Generator,
               eos_ids: set, pad_id: int, device, dtype):
    """Drive one batch of <= batch_size sequences to completion. Mutates states.
    Returns (n_target_forwards, n_draft_forwards, peak_mem_mb)."""
    n_tf = n_df = 0
    peak = 0.0
    while not all(s.finished for s in states):
        active = [i for i, s in enumerate(states) if not s.finished]
        # per-seq gamma clamp (remaining budget), exactly like speculative_generate
        g_of = {i: min(gamma, max_new_tokens - len(states[i].gen_ids)) for i in active}
        g_max = max(g_of.values())

        # ---- draft phase: g_max batched draft forwards; stash the exact rows -
        draft_tok = {i: [] for i in active}
        p_dm_rows = {i: [] for i in active}
        working = {i: states[i].full() for i in active}
        for k in range(g_max):
            batch_ids = [working[i] for i in active]
            inp, attn, pos = _pad_batch(batch_ids, pad_id, device, dtype)
            logits = _fwd(draft_model, inp, attn, pos)
            n_df += 1
            for bi, i in enumerate(active):
                if k >= g_of[i]:
                    continue
                row = dist_from_logits(logits[bi, -1, :], temperature)
                tok = _sample(row, generator)
                draft_tok[i].append(tok)
                p_dm_rows[i].append(row)
                working[i] = working[i] + [tok]

        # ---- target phase: one batched forward over prefix + drafts ---------
        batch_ids = [states[i].full() + draft_tok[i] for i in active]
        inp, attn, pos = _pad_batch(batch_ids, pad_id, device, dtype)
        tlogits = _fwd(target_model, inp, attn, pos)
        n_tf += 1
        peak = _peak_mb(peak)

        for bi, i in enumerate(active):
            s = states[i]
            g = g_of[i]
            actual_len = len(batch_ids[bi])
            pad_off = inp.shape[1] - actual_len
            ctx_len = len(s.full())
            # rows ctx_len-1 .. ctx_len-1+g  (row g is the bonus slot)
            p_tm_rows = [dist_from_logits(tlogits[bi, pad_off + ctx_len - 1 + j, :], temperature)
                         for j in range(g + 1)]
            n_acc = 0
            resampled = None
            for j in range(g):
                x = draft_tok[i][j]
                p_dm_x = float(p_dm_rows[i][j][x])
                p_tm_x = float(p_tm_rows[j][x])
                a = acceptance_probability(p_dm_x, p_tm_x)
                r = _uniform(generator)
                if r < a:
                    n_acc += 1
                    continue
                resampled = _sample(adjusted_distribution(p_dm_rows[i][j], p_tm_rows[j]), generator)
                break

            if n_acc == g:
                bonus = _sample(p_tm_rows[g], generator)
                emitted = draft_tok[i][:g] + [bonus]
            else:
                emitted = draft_tok[i][:n_acc] + [resampled]

            # in-block EOS truncation (坑13)
            hit_eos = False
            for kk, tid in enumerate(emitted):
                if tid in eos_ids:
                    emitted = emitted[:kk + 1]
                    hit_eos = True
                    break

            s.gen_ids.extend(emitted)
            s.accept_lengths.append(n_acc)
            s.emitted_per_round.append(len(emitted))
            s.accepted_total += n_acc
            s.evaluated_total += n_acc + (0 if n_acc == g else 1)
            s.rounds += 1
            if hit_eos or len(s.gen_ids) >= max_new_tokens:
                s.finished = True

        peak = _peak_mb(peak)
    return n_tf, n_df, peak


@torch.no_grad()
def speculative_generate_batch(prompts, draft_model, target_model, tokenizer, *,
                               gamma: int = 3, max_new_tokens: int = 64,
                               temperature: float = 1.0, seed: int = 0,
                               batch_size: int = 1,
                               apply_chat_template: bool = True) -> BatchGenResult:
    """Batched speculative decoding. `prompts` is a list of strings; they are
    processed in chunks of `batch_size`. batch_size=1 reproduces
    speculative_generate token-for-token at the same seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device
    dtype = torch.long
    eos_ids = collect_eos_ids(tokenizer, target_model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = next(iter(eos_ids), 0)

    all_states: List[SeqState] = []
    for p in prompts:
        ids = encode_prompt(tokenizer, p, device, apply_chat_template)
        all_states.append(SeqState(prompt_ids=[int(t) for t in ids[0].tolist()]))

    n_tf = n_df = 0
    peak = 0.0
    t0 = time.perf_counter()
    for start in range(0, len(all_states), batch_size):
        chunk = all_states[start:start + batch_size]
        c_tf, c_df, c_peak = _run_chunk(
            chunk, draft_model, target_model, gamma=gamma, max_new_tokens=max_new_tokens,
            temperature=temperature, generator=generator, eos_ids=eos_ids,
            pad_id=pad_id, device=device, dtype=dtype)
        n_tf += c_tf
        n_df += c_df
        peak = max(peak, c_peak)
    wall = time.perf_counter() - t0

    total_emitted = sum(len(s.gen_ids) for s in all_states)
    per_seq_rate = [len(s.gen_ids) / s.rounds for s in all_states if s.rounds]
    all_acc = [a for s in all_states for a in s.accept_lengths]
    acc_tot = sum(s.accepted_total for s in all_states)
    eval_tot = sum(s.evaluated_total for s in all_states)

    return BatchGenResult(
        texts=[tokenizer.decode(s.gen_ids, skip_special_tokens=True) for s in all_states],
        token_ids=[s.gen_ids for s in all_states],
        per_seq=[{"n_emitted": len(s.gen_ids), "rounds": s.rounds,
                  "accept_lengths": s.accept_lengths,
                  "alpha": (s.accepted_total / s.evaluated_total) if s.evaluated_total else 0.0}
                 for s in all_states],
        batch_size=batch_size,
        n_prompts=len(prompts),
        n_target_forwards=n_tf,
        n_draft_forwards=n_df,
        total_emitted=total_emitted,
        tokens_per_target_forward=total_emitted / n_tf if n_tf else 0.0,
        mean_tokens_per_seq_per_round=float(sum(per_seq_rate) / len(per_seq_rate)) if per_seq_rate else 0.0,
        mean_accept_length=float(sum(all_acc) / len(all_acc)) if all_acc else 0.0,
        alpha=(acc_tot / eval_tot) if eval_tot else 0.0,
        wall_s=wall,
        tok_per_s=total_emitted / wall if wall else 0.0,
        peak_mem_mb=peak,
        n_seq_finished_early=sum(1 for s in all_states if len(s.gen_ids) < max_new_tokens),
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_loader import load_model_and_tokenizer, DRAFT_MODEL_NAME, TARGET_MODEL_NAME
    from prompts import PROMPTS

    draft, tok = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, _ = load_model_and_tokenizer(TARGET_MODEL_NAME)
    for bs in (1, 2, 4):
        r = speculative_generate_batch(PROMPTS[:4], draft, target, tok,
                                       gamma=3, max_new_tokens=24, temperature=1.0,
                                       seed=0, batch_size=bs)
        print(f"bs={bs}: tok/target-fwd {r.tokens_per_target_forward:.2f}  "
              f"tok/seq/round {r.mean_tokens_per_seq_per_round:.3f}  alpha {r.alpha:.3f}  "
              f"tok/s {r.tok_per_s:.1f}  peak {r.peak_mem_mb:.0f}MB")
