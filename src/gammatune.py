"""
P5.0 -- GammaTune: an adaptive speculative-decoding step-size (gamma) controller.

Reference: Kim et al. 2025, "GammaTune" (arXiv 2504.00030); the algorithm and the
three-round worked example are transcribed from notes/project_plan_v9.md appendix
A.2 *verbatim* -- do not re-derive from memory.

Algorithm 1 -- run once after every speculative round. Inputs: `A` = drafts
actually accepted this round (== StepResult.n_accepted), `gamma` = the window size
used *entering* this round, `gamma_bar` = the running EMA estimate of the accept
length (a float carried across rounds).

    if A == gamma:                    # whole window accepted -> maybe headroom -> expand fast
        gamma <- clip(gmin, gmax, A + delta)
        # gamma_bar (the EMA) is NOT touched on this branch
    else:                             # not all accepted -> fall back to the conservative EMA
        gamma_bar <- clip(gmin, gmax, (1 - eta) * gamma_bar + eta * A)
        gamma <- ceil(gamma_bar)

Design intent (appendix A.2): "expand fast, contract slow" -- consecutive
full-accept rounds ratchet gamma up (3 -> 5 -> 7 -> ... -> gamma_max), but a
single less-than-full round drops straight back to the EMA so one unlucky round
cannot let gamma keep growing unchecked. P5.1 stresses exactly this behaviour in a
non-stationary prompt stream.

Hyper-parameters are the paper's defaults (eta=0.3, delta=2, gmin=1, gmax=10,
initial gamma_bar=3, gamma=3) and are treated as fixed constants here. They are
deliberately NOT tuned on this project's prompts: notes/project_plan_v9.md sec
9.6 risk 1 -- tuning on the same prompts later used to "validate" the controller
would only measure overfit. `GammaTuneConfig` exists so the values are easy to
change, not so this task changes them.

The raw appendix-A.2 listing writes the expand branch as `gamma <- A + delta`
with no clip; we clip it to [gmin, gmax] as well (the task spec is explicit, and
`A` can be gamma_max so `A + delta` can overshoot). The three-round example never
reaches the boundary, so both forms agree there.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from rejection_sampling import collect_eos_ids, encode_prompt, speculative_step


# --------------------------------------------------------------------------- #
# Pure controller update -- unit-tested directly against the appendix A.2 table
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def gammatune_update(
    gamma: int,
    gamma_bar: float,
    A: int,
    *,
    eta: float = 0.3,
    delta: int = 2,
    gmin: int = 1,
    gmax: int = 10,
) -> Tuple[int, float]:
    """One GammaTune step. Returns (new_gamma:int, new_gamma_bar:float).

    `A == gamma` is judged against the gamma that was in force *entering* the
    round (the caller's job to pass the right one), never against the freshly
    updated EMA.
    """
    if A == gamma:
        # expand branch: ratchet up, leave the EMA untouched
        new_gamma = int(_clip(A + delta, gmin, gmax))
        return new_gamma, gamma_bar
    # EMA branch: contract toward the conservative running estimate
    new_gamma_bar = _clip((1.0 - eta) * gamma_bar + eta * A, gmin, gmax)
    new_gamma = int(_clip(math.ceil(new_gamma_bar), gmin, gmax))
    return new_gamma, new_gamma_bar


@dataclass
class GammaTuneConfig:
    eta: float = 0.3
    delta: int = 2
    gmin: int = 1
    gmax: int = 10
    gamma_init: int = 3
    gamma_bar_init: float = 3.0


# --------------------------------------------------------------------------- #
# Full adaptive generation loop -- reuses speculative_step unchanged
# --------------------------------------------------------------------------- #
@dataclass
class GammaTuneResult:
    text: str
    token_ids: List[int]
    accept_lengths: List[int]        # n_accepted per round (0..gamma_that_round)
    emitted_per_round: List[int]     # tokens committed per round (n_accepted + 1, EOS-truncated)
    gamma_trace: List[int]           # gamma in force *entering* each round
    n_rounds: int
    elapsed_s: float
    final_state: Tuple[int, float]   # (gamma, gamma_bar) after the last round -- feed to carry_state
    mean_emitted_per_round: float    # PRIMARY metric: tokens per target forward
    mean_accept_length: float
    proposals: List[dict] = field(default_factory=list)


@torch.no_grad()
def gammatune_generate(
    prompt: str,
    draft_model,
    target_model,
    tokenizer,
    *,
    config: GammaTuneConfig = GammaTuneConfig(),
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    seed: int = 0,
    carry_state: Optional[Tuple[int, float]] = None,
    record: bool = False,
    apply_chat_template: bool = True,
) -> GammaTuneResult:
    """Speculative decoding with GammaTune choosing gamma each round.

    Each round: run one `speculative_step` at the current gamma, then feed
    `n_accepted` through `gammatune_update` to pick the next gamma. Rejection
    sampling itself is untouched (`speculative_step` from P1.1).

    `carry_state=(gamma, gamma_bar)` seeds the controller from a previous call so
    it does NOT reset at every prompt boundary -- P5.1 needs the controller to
    stay warm across a non-stationary prompt stream. `carry_state=None` starts
    from the config's initial values.

    坑13 (notes/project_plan_v9.md sec 9.2): a round commits "accepted prefix + 1
    token" as one chunk and an EOS can land in the middle of it. The chunk is
    scanned for the first EOS and truncated there, same as `speculative_generate`
    -- without this, speculative output runs past EOS and diverges from a
    token-by-token baseline.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device

    context = encode_prompt(tokenizer, prompt, device, apply_chat_template)
    eos_ids = collect_eos_ids(tokenizer, target_model)

    if carry_state is not None:
        gamma, gamma_bar = int(carry_state[0]), float(carry_state[1])
    else:
        gamma, gamma_bar = int(config.gamma_init), float(config.gamma_bar_init)

    token_ids: List[int] = []
    accept_lengths: List[int] = []
    emitted_per_round: List[int] = []
    gamma_trace: List[int] = []
    proposals: List[dict] = []

    t0 = time.perf_counter()
    while len(token_ids) < max_new_tokens:
        gamma_trace.append(gamma)
        # Cap only the actual step to the remaining token budget; the controller
        # still sees its own gamma. A capped step can only be the final round
        # (the loop exits on the budget), so the update it produces is unused.
        g_eff = min(gamma, max_new_tokens - len(token_ids))
        step = speculative_step(
            context,
            draft_model,
            target_model,
            g_eff,
            temperature=temperature,
            generator=generator,
            record=record,
        )

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
        if record:
            proposals.extend(step.proposals)
        context = torch.cat(
            [context, torch.tensor([emitted], device=device, dtype=context.dtype)], dim=1
        )

        gamma, gamma_bar = gammatune_update(
            gamma, gamma_bar, step.n_accepted,
            eta=config.eta, delta=config.delta, gmin=config.gmin, gmax=config.gmax,
        )
        if hit_eos:
            break
    elapsed = time.perf_counter() - t0

    n_rounds = len(accept_lengths)
    return GammaTuneResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        accept_lengths=accept_lengths,
        emitted_per_round=emitted_per_round,
        gamma_trace=gamma_trace,
        n_rounds=n_rounds,
        elapsed_s=elapsed,
        final_state=(gamma, gamma_bar),
        mean_emitted_per_round=(sum(emitted_per_round) / n_rounds) if n_rounds else 0.0,
        mean_accept_length=(sum(accept_lengths) / n_rounds) if n_rounds else 0.0,
        proposals=proposals,
    )
