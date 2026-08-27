"""
P5.3 -- Batch-aware circuit breaker for speculative decoding.

Reference: notes/project_plan_v9.md sec 7 P5.3 (lines 184-186) + sec 9.2 pitfalls
7 / 11 / 12 + sec 9.6 risk 4 + sec 13 walkthrough (lines 344-348).

What this answers, and how it differs from GammaTune (P5.0):
  GammaTune picks *how big* the speculation window gamma should be. The circuit
  breaker picks *whether to speculate at all*. They are orthogonal. Production
  consensus (Nightjar, sec 9.2 pitfall 7: up to 30.25% throughput regression
  under load): once the batch is large the target model already saturates the
  accelerator, so speculation no longer hides the target's per-step latency --
  the draft forwards become pure overhead. The breaker watches a batch-size
  signal and degrades to plain target autoregression when the batch is high.

Three mechanisms the plan requires:
  1. Threshold trip (pitfall 7)      -- batch_size >= batch_threshold -> degrade.
  2. Periodic re-probe (pitfall 11)  -- Nightjar's critique of DSD is "once
     disabled it never collects data again and cannot restart". While degraded,
     every `reprobe_every` steps we force one small-gamma speculative round and
     record its alpha / accept length *per task*, not as one global number
     (sec 13, lines 344-348: a batch of tasks whose alpha dropped because the
     task *type* changed, not because the batch rose -- only per-task records
     let a controller tell those apart). When the batch falls back below the
     threshold we re-probe once before committing back to speculation.
  3. Measured switch cost (pitfall 12) -- BanditSpec is criticised for assuming
     switching is free. This project has no KV cache, so there is nothing to
     rebuild; instead we measure the *proxy* cost of the wasted work a switch
     forces: re-running the full current prefix through both models (the work a
     KV-cache handoff would otherwise save). Labelled a proxy in the results,
     not a real KV-cache rebuild number. Nightjar's own measured range on
     RTX 4090 + 7B is 17.87ms (short/small) to 102.03ms (long/large).

Primary metric (pitfall 14 -- do NOT use mean_emitted_per_round here): it has no
draft-cost term, is monotone in gamma, and cannot express "speculate vs not".
Use cost-model throughput  total_emitted / total_cost_units  where a speculative
round costs (c + gamma) and a degraded target step costs c
(c = T_target / T_draft, from verify_gammatune.measure_c; also reported for the
literature range c in {4, 7, 10} since the local MPS c is launch-overhead bound).

The batch-size signal is an externally injected synthetic trace (e.g.
[1,1,4,8,32,32,8,1,...]); there is no real high-batch throughput data on this
machine. Wiring a real batch curve in is M5[A] / cloud.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from rejection_sampling import (
    _sample,
    collect_eos_ids,
    dist_from_logits,
    encode_prompt,
    speculative_step,
)


# --------------------------------------------------------------------------- #
# Config -- thresholds are frozen defaults with a stated rationale, NOT tuned
# on this project's prompt set (sec 9.6 risk 1: a threshold fit on these prompts
# and then "validated" on similar prompts only measures overfit).
# --------------------------------------------------------------------------- #
@dataclass
class CircuitBreakerConfig:
    # batch_size >= this -> degrade to target-only. 8 is a deliberately
    # conservative mid value: P1.4 put the speculative cost/benefit knee for this
    # model pair around gamma=3, and the production reports (pitfall 7) put the
    # regression onset in the "batch > a few tens" range; 8 trips early rather
    # than waiting for a measured regression this machine cannot produce.
    batch_threshold: int = 8
    # while degraded, force one speculative probe every N generation rounds
    # (pitfall 11). 50 rounds ~ a few full generations at spec_gamma=3.
    reprobe_every: int = 50
    # gamma for a probe round -- small so a probe that lands in a genuinely bad
    # regime wastes little draft compute.
    reprobe_gamma: int = 3
    # gamma for a normal speculative round (P1.4 cost/benefit knee).
    spec_gamma: int = 3


# --------------------------------------------------------------------------- #
# Pure decision function -- this is the core of the unit tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CircuitBreakerState:
    mode: str = "spec"               # underlying generation mode: "spec" | "target"
    last_reprobe_step: int = -10**9  # round index the re-probe clock was last reset
    just_reprobed: bool = False      # was the immediately preceding round a probe?


@dataclass
class Decision:
    mode: str                        # what to run THIS round: "spec" | "target" | "reprobe"
    switched: bool                   # did the underlying generation mode flip?
    direction: Optional[str]         # "spec->target" | "target->spec" | None
    reason: str


def circuit_breaker_decide(
    state: CircuitBreakerState,
    batch_size: int,
    step: int,
    config: CircuitBreakerConfig = CircuitBreakerConfig(),
) -> Decision:
    """Decide what the given round should run, from the current state and the
    injected batch size. Pure: does not mutate `state`; the driver calls
    `advance_state` with the returned Decision.

    "reprobe" is a speculative round at `reprobe_gamma` whose alpha is logged
    per task; the underlying mode stays "target" until the batch actually clears.
    """
    high = batch_size >= config.batch_threshold

    if state.mode == "spec":
        if high:
            return Decision("target", True, "spec->target", "batch>=threshold: degrade")
        return Decision("spec", False, None, "batch<threshold: keep speculating")

    # state.mode == "target" (degraded)
    if high:
        if step - state.last_reprobe_step >= config.reprobe_every:
            return Decision("reprobe", False, None,
                            "periodic probe while degraded (pitfall 11)")
        return Decision("target", False, None, "batch>=threshold: stay degraded")

    # batch has fallen back below threshold while degraded -> recover, but probe
    # once first unless the immediately preceding round was already a probe.
    if not state.just_reprobed:
        return Decision("reprobe", False, None,
                        "recovery probe after batch drop (pitfall 11)")
    return Decision("spec", True, "target->spec",
                    "batch<threshold and probe done: restore speculation")


def advance_state(state: CircuitBreakerState, decision: Decision, step: int) -> CircuitBreakerState:
    """Fold a Decision back into the state for the next round.

    The re-probe clock is reset both on an actual probe round and on a fresh
    spec->target degrade, so the first periodic probe fires `reprobe_every`
    rounds *after* degrading rather than immediately.
    """
    mode = decision.mode if decision.mode in ("spec", "target") else state.mode
    if decision.mode == "reprobe" or decision.direction == "spec->target":
        last = step
    else:
        last = state.last_reprobe_step
    return CircuitBreakerState(
        mode=mode, last_reprobe_step=last, just_reprobed=(decision.mode == "reprobe")
    )


def simulate_decisions(
    config: CircuitBreakerConfig,
    batch_trace: List[int],
    *,
    start: Optional[CircuitBreakerState] = None,
) -> List[Decision]:
    """Run the state machine over a batch-size trace with no models attached.
    Pure and fully deterministic -- the unit tests feed it a synthetic trace and
    assert the degrade / probe / restore steps."""
    state = start or CircuitBreakerState()
    out: List[Decision] = []
    for step, bs in enumerate(batch_trace):
        d = circuit_breaker_decide(state, bs, step, config)
        out.append(d)
        state = advance_state(state, d, step)
    return out


# --------------------------------------------------------------------------- #
# Switch-cost probe (pitfall 12, proxy variant)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_switch_cost(draft_model, target_model, context: torch.Tensor, reps: int = 10) -> dict:
    """Proxy cost of one mode switch: with no KV cache, a switch forces the next
    round to reprocess the whole current prefix from scratch through both models
    (exactly the work a KV-cache handoff would save). Time that, in ms.

    This is a PROXY for the KV-cache rebuild cost, not the real thing -- stated
    as such in the results file. Nightjar's real measured range (RTX 4090 + 7B)
    is 17.87-102.03 ms; a small model pair on MPS will sit well below that.
    """
    for _ in range(3):  # warm-up
        draft_model(context)
        target_model(context)
    td = tt = 0.0
    for _ in range(reps):
        t0 = time.perf_counter(); draft_model(context); td += time.perf_counter() - t0
        t0 = time.perf_counter(); target_model(context); tt += time.perf_counter() - t0
    prefix_len = int(context.shape[1])
    return {
        "proxy": "re-encode full prefix through both models (no KV cache to hand off)",
        "prefix_len": prefix_len,
        "draft_forward_ms": td / reps * 1e3,
        "target_forward_ms": tt / reps * 1e3,
        "switch_cost_ms": (td + tt) / reps * 1e3,
        "reps": reps,
        "nightjar_reference_ms": [17.87, 102.03],
    }


# --------------------------------------------------------------------------- #
# Full generation loop
# --------------------------------------------------------------------------- #
@dataclass
class CircuitBreakerResult:
    texts: List[str]
    per_round_mode: List[str]          # "spec" | "target" | "reprobe" for every round, all prompts
    per_round_batch: List[int]         # the injected batch size seen at each round
    per_round_gamma: List[int]         # gamma the round ran at (0 for a target step)
    per_round_emitted: List[int]       # tokens committed by each round
    mode_switches: List[dict]          # {step, prompt_index, direction, switch_cost_ms}
    reprobe_log: List[dict]            # {step, prompt_index, gamma, alpha, mean_accept_len, n_rounds}
    emitted_total: int
    cost_units_total: dict             # {"measured": x, "c4": x, "c7": x, "c10": x}
    spec_rounds: int
    target_rounds: int
    reprobe_rounds: int
    n_rounds: int
    elapsed_s: float
    switch_cost_probe: dict
    final_state: Tuple[str, int]


def _cost_units(c: float, spec_rounds_gamma_sum: int, reprobe_rounds_gamma_sum: int,
                target_rounds: int, n_spec_rounds: int, n_reprobe_rounds: int) -> float:
    """Cost model (pitfall 14): a speculative/probe round costs (c + gamma), a
    degraded target step costs c."""
    return ((n_spec_rounds + n_reprobe_rounds) * c
            + spec_rounds_gamma_sum + reprobe_rounds_gamma_sum
            + target_rounds * c)


@torch.no_grad()
def circuit_breaker_generate(
    prompts: List[str],
    draft_model,
    target_model,
    tokenizer,
    *,
    config: CircuitBreakerConfig = CircuitBreakerConfig(),
    batch_size_trace: List[int],
    max_new_tokens: int = 48,
    temperature: float = 1.0,
    seed: int = 0,
    measured_c: float = 0.0,
    apply_chat_template: bool = True,
) -> CircuitBreakerResult:
    """Generate every prompt in `prompts`, keeping ONE circuit-breaker state
    across the whole stream (like GammaTune's carry_state). The batch-size trace
    is consumed by a global round counter shared across prompts; when it runs out
    the last value is held.

    Speculative rounds reuse `speculative_step` at `config.spec_gamma`; degraded
    steps run one plain target forward; probe rounds run `speculative_step` at
    `config.reprobe_gamma` and log alpha per prompt. Block-internal EOS is
    truncated per pitfall 13.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    device = next(target_model.parameters()).device
    eos_ids = collect_eos_ids(tokenizer, target_model)

    if not batch_size_trace:
        raise ValueError("batch_size_trace must be non-empty")

    state = CircuitBreakerState()
    texts: List[str] = []
    per_round_mode: List[str] = []
    per_round_batch: List[int] = []
    per_round_gamma: List[int] = []
    per_round_emitted: List[int] = []
    mode_switches: List[dict] = []
    reprobe_log: List[dict] = []
    emitted_total = 0
    spec_gamma_sum = 0
    reprobe_gamma_sum = 0
    n_spec = n_target = n_reprobe = 0
    global_step = 0

    switch_probe: dict = {}

    t0 = time.perf_counter()
    for p_idx, prompt in enumerate(prompts):
        context = encode_prompt(tokenizer, prompt, device, apply_chat_template)
        token_ids: List[int] = []

        while len(token_ids) < max_new_tokens:
            bs = batch_size_trace[min(global_step, len(batch_size_trace) - 1)]
            decision = circuit_breaker_decide(state, bs, global_step, config)

            if decision.switched:
                if not switch_probe:
                    switch_probe = measure_switch_cost(draft_model, target_model, context)
                mode_switches.append({
                    "step": global_step,
                    "prompt_index": p_idx,
                    "direction": decision.direction,
                    "switch_cost_ms": switch_probe["switch_cost_ms"],
                })

            budget = max_new_tokens - len(token_ids)

            round_gamma = 0
            if decision.mode in ("spec", "reprobe"):
                g = config.spec_gamma if decision.mode == "spec" else config.reprobe_gamma
                g_eff = min(g, budget)
                round_gamma = g_eff
                step = speculative_step(
                    context, draft_model, target_model, g_eff,
                    temperature=temperature, generator=generator,
                )
                emitted = step.new_token_ids
                if decision.mode == "spec":
                    n_spec += 1
                    spec_gamma_sum += g_eff
                else:
                    n_reprobe += 1
                    reprobe_gamma_sum += g_eff
                    reprobe_log.append({
                        "step": global_step,
                        "prompt_index": p_idx,
                        "gamma": g_eff,
                        "alpha": (step.n_accepted / step.n_evaluated) if step.n_evaluated else 0.0,
                        "n_accepted": step.n_accepted,
                        "n_evaluated": step.n_evaluated,
                    })
            else:  # "target": one plain target forward
                row = dist_from_logits(target_model(context).logits[0, -1, :], temperature)
                tok = _sample(row, generator)
                emitted = [tok]
                n_target += 1

            # pitfall 13: EOS may land mid-block; truncate at (and keep) first EOS
            hit_eos = False
            for k, tid in enumerate(emitted):
                if tid in eos_ids:
                    emitted = emitted[: k + 1]
                    hit_eos = True
                    break

            token_ids.extend(emitted)
            emitted_total += len(emitted)
            per_round_mode.append(decision.mode)
            per_round_batch.append(int(bs))
            per_round_gamma.append(round_gamma)
            per_round_emitted.append(len(emitted))
            context = torch.cat(
                [context, torch.tensor([emitted], device=device, dtype=context.dtype)], dim=1
            )
            state = advance_state(state, decision, global_step)
            global_step += 1
            if hit_eos:
                break

        texts.append(tokenizer.decode(token_ids, skip_special_tokens=True))

    elapsed = time.perf_counter() - t0

    c_values = {"measured": measured_c, "c4": 4.0, "c7": 7.0, "c10": 10.0}
    cost_units_total = {
        k: _cost_units(c, spec_gamma_sum, reprobe_gamma_sum, n_target, n_spec, n_reprobe)
        for k, c in c_values.items()
    }

    return CircuitBreakerResult(
        texts=texts,
        per_round_mode=per_round_mode,
        per_round_batch=per_round_batch,
        per_round_gamma=per_round_gamma,
        per_round_emitted=per_round_emitted,
        mode_switches=mode_switches,
        reprobe_log=reprobe_log,
        emitted_total=emitted_total,
        cost_units_total=cost_units_total,
        spec_rounds=n_spec,
        target_rounds=n_target,
        reprobe_rounds=n_reprobe,
        n_rounds=len(per_round_mode),
        elapsed_s=elapsed,
        switch_cost_probe=switch_probe,
        final_state=(state.mode, state.last_reprobe_step),
    )
