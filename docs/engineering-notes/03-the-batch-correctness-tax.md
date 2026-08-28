# The batch correctness tax: measure the bill you chose not to pay

**Task:** P6.1 — batched / continuously-batched speculative decoding for a
serving loop.
**Pitfall:** 坑19.

## The problem batching creates for speculative decoding

Autoregressive batching is easy: every sequence advances one token per step, so
the batch stays rectangular. Speculative decoding breaks that. In one round,
sequence A accepts 4 draft tokens, sequence B accepts 1, sequence C rejects
immediately and resamples. Now the three sequences have different lengths,
different KV-cache fill levels, and different next-position indices. Any shared
tensor across the batch is ragged from round one.

The published fixes (EQSPEC, arXiv:2510.22876 and relatives) keep the batch
rectangular with padding + masking + per-round realignment, and the paper's own
point is that doing this *wrong* silently changes outputs — the batched result
stops matching what each sequence would have produced alone.

## The choice: sidestep the ragged tensor entirely

Instead of building a correct masked-and-realigned ragged-verify kernel, P6.1
gives **each sequence its own KV cache** and defines a "batch round" as: run the
already-verified single-sequence step ([note 02](02-getting-the-kv-cache-right.md))
once per active sequence.

The payoff is that **output equivalence holds by construction**. There is no
shared tensor to desync. A test pins `batched(prompts) == [single(p) for p in
prompts]`, bit-exact, in both greedy and sampling mode. No masking logic to get
subtly wrong, no "did the batch change the answer" anxiety.

That property has a name — **batch invariance** (Thinking Machines, *Defeating
Nondeterminism in LLM Inference*, 2025). The determinism literature earns it at
the *kernel* layer with batch-invariant RMSNorm / matmul / attention. This design
gets it for free at the *loop* layer by refusing to share a tensor in the first
place — a much cheaper move that only works because there's no kernel-level
batching to want. Oracle **O5** in [note 06](06-testing-a-speculative-decoder.md)
is the regression guard for it: it re-checks batch == N × single under every
fault-injection mutant, and the reference path stays bitwise.

The price is that there's no kernel-level batching of the verify forward, so
throughput barely moves with concurrency. Measured aggregate tok/s across
widths 1/2/4/8:

| regime | w1 | w2 | w4 | w8 | vs target-only-KV |
|--------|----|----|----|----|-------------------|
| short  | 24.9 | 24.9 | 24.4 | 25.0 | 0.99–1.03× |
| long   | 16.6 | 16.9 | 17.0 | 16.9 | 0.96–0.99× |

Flat. **That flatness is the finding**, not a disappointment to hide: on this
hardware, with per-sequence caches, you get correctness-by-construction and you
do not get a batch speedup. If you want the speedup you have to go build the
ragged-verify kernel and take on its correctness burden.

## Measuring the tax you didn't pay

"We avoided the realignment cost" is only a real claim if you can say how big it
was. So the harness computes it analytically, per round, against the rectangular
counterfactual an EQSPEC-style padded batch would have run:

```
realignment_overhead = 1 − (Σ_i work_i) / (n · max_i work_i)
```

i.e. the fraction of a padded batch's compute that would have been spent on
padding rows, given the actual per-sequence accept lengths that round.

- At **width 4** it peaks: mean **0.073** (short) / **0.096** (long), p90
  **0.68 / 0.74**. Some rounds, a padded batch would be two-thirds padding.
- At **width 8** it collapses back to **0.007–0.02**: when every sequence runs
  every round, per-round work is much more uniform, so a rectangular batch
  wastes little.

The tax is worst at **medium** concurrency — few enough sequences that one long
acceptance skews the round, many enough that padding to the max is expensive.
That's a non-obvious shape and it's the kind of thing you only learn by
computing the counterfactual instead of hand-waving it.

## The lesson

When you take the correct-by-construction path and skip an optimization, quantify
the optimization you skipped — as a measurement in the result file, on the same
workload. "Batch speedup isn't free" is a slogan; "a padded batch would be up to
74% padding at width 4, dropping to 2% at width 8" is a number a reader can act
on.
