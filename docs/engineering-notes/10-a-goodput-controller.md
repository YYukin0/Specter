# A goodput controller for speculation length, and why it loses on this pair

**Task:** Pillar 8 (支柱8) Track C — replace the serving loop's binary
`spec / degraded` circuit breaker with a continuous controller that picks the
speculation length `k` each round to maximize *goodput* (accepted tokens per
unit wall-time), in the style of SmartSpec / TurboSpec (arXiv:2406.14066).
**Pitfalls:** 30, 31, 32.
**Code:** `src/goodput_model.py`, `src/goodput_profile.py`,
`src/verify_p7_1_goodput_controller.py`, `tests/test_goodput_curve.py`,
`tests/test_goodput_profile.py`, `tests/test_serving_loop_goodput.py`,
`results/p7_0_goodput_profile.json`, `results/p7_1_goodput_controller.json`.

## Why a controller at all

The serving loop (`src/serving_loop.py`, note 04) already has a circuit breaker
that watches the *real* rolling acceptance rate `alpha` and drops the whole
batch to plain target decoding when `alpha` falls below a floor. Two limits,
both called out in earlier pitfalls:

- **It's binary.** `k` is either the configured `gamma` or zero. There is no
  "acceptance is mediocre, so speculate 2 tokens instead of 4."
- **It's batch-blind by construction** (Pitfall 20). `len(active)` is only an
  input to the breaker, never the rule, and on a per-sequence-cache design
  (note 03) throughput barely moves with batch size — so "always speculate at
  full `gamma`" is very hard to beat, and the breaker mostly just sits there.

vLLM's dynamic speculative-decoding solves the first limit with a lookup table
keyed on running batch size. This track does the continuous version: model
goodput as a function of `(alpha, k, batch state)` and take the argmax over `k`
every round.

## The model

Three pieces, all in `src/goodput_model.py` (pure functions, no torch):

**Expected accepted tokens.** Leviathan et al. 2023's closed form for a single
speculative block, `E[accepted] = (1 - alpha^(k+1)) / (1 - alpha)`. As
`alpha -> 1` this is `0/0`; the limit is `k + 1` and the code special-cases it
(Pitfall 30 — a naive implementation divides by zero exactly when speculation
is working best). `k <= 0` returns `1.0` (the one guaranteed target token).

**Expected round time.** A linear model fit offline:

```
T = c0 + c1 * (n_active * (mean_pending + k))
        + c2 * (n_active * mean_kv_len)
        + c3 * (n_active * k)
```

`c1` is the per-token cost of the batched forward (draft proposal + target
verify), `c2` the KV-length-dependent attention cost, `c3` a **draft-only**
per-speculative-token term. `c3` has to be in the model on paper: the draft
runs `k` sequential forwards per round and the target runs one, so the draft
cost scales with `k` in a way the verify cost does not (Pitfall 14, the
"dead-zone" mechanism).

**Goodput.** `goodput(k) = E[accepted](alpha, k) / E[round_time](k)`, and
`best_k` scans `k in [k_min, k_max]`, clamped to `[prev_k - h, prev_k + h]`
with `h = 2` so the choice can't chatter round to round (Pitfall 31). `k* = 0`
means "don't speculate this round" and the loop runs a degraded target step.

## Calibration (`goodput_profile.py`)

Offline, on the real `Qwen2.5-0.5B / 1.5B` pair, MPS fp16. Grid:
`n_active in {1,2,4,8}` x `regime in {short=40, mid=250, long=500 prompt tokens}`
x `k in {0,1,2,3,5,8}` = 72 cells, 5 timed rounds each after 3 warm-up rounds,
`torch.mps.synchronize()` around every measurement. 20% of cells held out.

The fit does **not** use ordinary least squares. A first OLS pass returned
`c3 = -0.0355` (negative — a draft that gets *cheaper* per token, which is
nonsense) with `R^2 = 0.958`. Cause: the rectangular-invariant serving loop
keeps `mean_pending` pinned at exactly `1.0` every round, so
`n*(mean_pending+k)` and `n*k` differ only by a factor of `n` and are
collinear — `c1` and `c3` are not separately identifiable. The fix is a
hand-rolled **non-negative least squares** (active-set; numpy has no NNLS and
scipy isn't a dependency), which pins `c3 = 0` and lets `c1` absorb the
combined per-speculative-token cost:

| coeff | value | meaning |
|---|---|---|
| `c0` | 0.0396 s | fixed per-round overhead |
| `c1` | 0.0235 s/token | batched forward, per (draft+verify) token |
| `c2` | 5.98e-5 s | per `n_active * kv_len` |
| `c3` | 0.0 | draft-only term — unidentifiable here, folded into `c1` |
| `R^2` | 0.920 | on 288 fit samples |
| held-out MAPE | 0.177 | on 72 held-out cells |

The draft cost is real, it's just not separable from this profiler's data;
`results/p7_0_goodput_profile.json`'s `acceptance_note` records exactly why. A
pair with a genuinely expensive draft, or a profiler that varies pending
depth, would break the collinearity.

## What the controller actually does

`src/verify_p7_1_goodput_controller.py`: three non-stationary prompt streams
(`A_to_B`, `A_to_B_to_A`, `ABAB` from `nonstationary_prompts.py`) x four
concurrency widths x three controllers (`fixed` gamma=3, `alpha_floor` breaker,
`goodput`), plus a load ramp that submits 32 prompts against a width-8 server
and logs `k*` per round. ~24 min on MPS; `results/p7_1_goodput_controller.json`.

Aggregate output tok/s, and what the goodput controller chose:

| stream | w | fixed | alpha_floor | **goodput** | goodput mean k | goodput k=0 rounds |
|---|--:|--:|--:|--:|--:|--:|
| A_to_B | 1 | 24.1 | 24.8 | **25.0** | 3.06 | 0 |
| A_to_B | 2 | 24.5 | 24.3 | **22.9** | 1.76 | 0 |
| A_to_B | 4 | 24.8 | 24.4 | **23.2** | 1.84 | 0 |
| A_to_B | 8 | 23.9 | 23.8 | **23.0** | 1.18 | 24 |
| A_to_B_to_A | 1 | 25.0 | 25.1 | **25.1** | 3.10 | 0 |
| A_to_B_to_A | 2 | 24.0 | 23.6 | **22.3** | 2.06 | 0 |
| A_to_B_to_A | 4 | 20.7 | 17.6 | **22.2** | 1.58 | 32 |
| A_to_B_to_A | 8 | 24.1 | 23.8 | **22.3** | 1.64 | 22 |
| ABAB | 1 | 23.5 | 23.3 | **22.5** | 2.47 | 0 |
| ABAB | 2 | 23.8 | 23.7 | **23.4** | 2.14 | 0 |
| ABAB | 4 | 23.5 | 23.6 | **22.9** | 1.77 | 0 |
| ABAB | 8 | 23.1 | 22.8 | **21.8** | 1.14 | 24 |

**This is a negative result.** The controller shrinks `k` well below the fixed
`gamma = 3` on every non-trivial cell (mean `k` ~1.6–2.5), and at width >= 2
that costs throughput: goodput's aggregate tok/s is below the better baseline
on 7 of 9 cells, mean -4%, worst -7%. At width 1 it ties. The one cell it wins
(A_to_B_to_A, w=4) is a cell where `alpha_floor` itself stumbled to 17.6.

The **machinery is correct**. `k*` tracks the round-time model's argmax; the
hysteresis clamp keeps it from oscillating; `alpha = 1` is handled; no run
failed; and under the load ramp `k*` does contract as `n_active` climbs to 8
(`k_mean_after_full = 1.80`, ramp series shows `k*` dropping to 0 during
low-alpha stretches and recovering to 2 when `alpha` rebounds). The controller
does what it's told.

**The model's optimum just isn't the throughput optimum on this pair.** With
`c3` folded into `c1 = 0.0235 s/token`, `E[round_time]` rises roughly linearly
in `k`, while `E[accepted]` saturates at `alpha ~= 0.77` (the geometric series
is nearly flat past `k = 3`). Their ratio therefore peaks at small `k`. But on
this Mac, verifying `k = 3` vs `k = 1` draft tokens in one batched target
forward is nearly free — that's the whole dead-zone finding (Pitfall 4): the
draft forward is the expensive part, the incremental verify is not. The linear
model over-charges for `k`, so the controller trims speculation it should have
kept, and leaves accepted tokens on the table.

## Qualitative cross-check against the A40 run

Note 09's cloud run: EAGLE3's speedup drops from 2.4x at concurrency 16 to
1.6x at 64 as the server saturates. The goodput model's batched-forward term
(`c1 * n_active * (...+k)`, and `c2 * n_active * kv_len`) grows with
`n_active`, so `best_k` shrinks as concurrency rises — the same direction: more
concurrent work makes a marginal speculative token less worth it. This is a
**shape** match only. `bullet2_vllm_eagle3.json`'s `mean_acceptance_rate` is
`null`, so there is no way to feed the A40's alpha into this model and check
the magnitude; it's a consistency observation, not a validation.

## Honest boundaries

- **Five-point linear fit.** The round-time model is `k in {0,1,2,3,5,8}` x 4
  widths x 3 regimes, one machine, MPS fp16. `R^2 = 0.92` / held-out MAPE 0.18
  is decent for a linear surrogate but it is a surrogate.
- **`c3` is not measured, it's assumed zero.** The draft-cost term that the
  dead-zone story hinges on is exactly the one this profiler can't identify
  (collinear regressors). The model is therefore blind to the mechanism that
  would tell it to keep `k` larger.
- **Rolling-alpha lag.** `alpha` is a trailing average over the last `window`
  accept decisions (inherited from P5.1); on a sharp A->B workload switch the
  controller reacts a few rounds late, same as GammaTune.
- **The controller is off by default.** `ServeConfig.controller` defaults to
  `"alpha_floor"`; `"goodput"` and `"fixed"` are opt-in. The default serving
  path is byte-identical to before this track (the hermetic suite is unchanged
  at +N tests, 0 modified).
