# The circuit breaker: a real signal, and a premise that had gone stale

**Task:** P5.3 (first cut) → P6.1 (rebuild).
**Pitfalls:** 坑15, 坑20.

## The original idea

Speculative decoding can be *slower* than plain decoding when the draft and
target disagree a lot (every rejected token is wasted draft compute) or when the
accelerator is saturated and the extra draft forwards just add contention. So:
add a circuit breaker that watches for those conditions and falls back to
target-only decoding when speculation stops paying off.

P5.3 built that breaker with a **synthetic** batch-load signal — there is no real
high-batch throughput on this single Mac, so batch pressure was injected
externally — and scored it on a cost-model throughput metric:

```
throughput = total_emitted_tokens / total_cost_units
   spec round      costs  c + γ
   degraded step   costs  c
```

## Why that couldn't be evaluated honestly

**No term in that formula depends on batch size.** A speculative round always
emits `≥ 1` token for its `c + γ` cost; a degraded step emits exactly 1 for `c`.
So "always speculate" is a structural upper bound the breaker can only *tie*,
never beat — degrading always does less useful work per unit compute, and there's
no batch-contention penalty in the model to make speculation look bad. The
synthetic signal was never wired back into anything the metric could see.

Measured, as expected: always-spec **0.280 ± 0.010** vs breaker
**0.247 ± 0.008**. The headline metric said the breaker was strictly harmful.
That's not a real result — it's an artifact of scoring a batch-aware mechanism
with a batch-blind metric on a machine that can't produce the pressure the
mechanism exists for. (A saturation-aware "Metric B" with a `sat_tax ∈ {1,2,3}`
contention penalty was added as a sensitivity check and roughly reproduces
Nightjar's regression order of magnitude — but it's a sensitivity knob, not
measured hardware behaviour, and it was kept clearly labelled as such rather than
promoted to the headline.)

## The rebuild (P6.1)

The breaker now trips on signals that are **actually observable on this box**:

- A **real rolling acceptance rate** α: the windowed mean of the last
  `alpha_window` accept/reject decisions across all active sequences. Trip when
  α `< alpha_floor`.
- Optionally, a **target-only latency probe**: if a periodic measurement shows
  speculative rounds are wall-clock slower than plain decoding right now, trip.

`len(active_sequences)` is an **input** to the policy, never the rule by itself.
And while degraded, the loop forces a speculative **probe round** every
`reprobe_every` rounds so α stays measurable — otherwise a breaker that's
disabled speculation has no data to decide to re-enable it (坑11: DSD-style
breakers that "stop collecting when off" can never turn back on).

## What it actually did

Over 16 serving runs (2 regimes × 4 widths × breaker on/off), the breaker
tripped **exactly once** — `long` regime, width 2 — and **never** because of
batch size (width 8 never degrades at all). In that one run it logged 20 degraded
rounds and 1 probe round, and aggregate throughput was within noise of
breaker-off (17.1 vs 16.9 tok/s).

That looks like a null result. It isn't a *failure*: on this model pair α sits
around 0.75–0.80 the whole time, which is healthy, so a correctly-built breaker
**should** be a near-no-op. "Never degrades under a healthy α" is the right
behaviour. What the exercise actually surfaced is that you cannot *validate* a
breaker on a workload where the acceptance rate never drops — you need a
deliberately mismatched draft/target pair or an adversarial prompt distribution
to drive α into the floor, and that's a separate experiment.

## The lesson

Two distinct traps, one mechanism:

1. **Don't score a mechanism with a metric that structurally can't reward it.**
   If the cost model has no term for the thing the feature responds to, the
   feature will always look useless — and that's a statement about the metric,
   not the feature.
2. **A guard rail you can't trip is a guard rail you haven't tested.** Build the
   breaker on a real signal, then go find (or construct) the regime where that
   signal actually goes bad. Under normal conditions it should do nothing, and
   that's fine — but "does nothing on every workload I tried" means the
   validation workload was wrong.
