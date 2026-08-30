# Mac vs. rented A40: the same break-even story, at a very different price

**Task:** Bullet 2 (Pillar 7) — a one-time rented-GPU run of the same
speculative-decoding question this repo asks on Apple silicon, against a
production stack: vLLM 0.10.2, real EAGLE3 head, real concurrency, a real
GSM8K workload.
**Pitfalls:** 27, 28, 29.
**Code:** `src/cloud_bench/` (`config.py`, `orchestrate.py`, `sanity_check.py`),
`tests/test_cloud_bench.py`, `results/bullet2_vllm_eagle3.json`,
`results/cloud_bench_raw/`.

## Why this comparison

Everything else in this repo runs on a 24 GB M-series Mac, a draft/target pair
picked to fit in unified memory (`Qwen2.5-0.5B/1.5B-Instruct`), and a
from-scratch decode loop. The headline finding on that setup is **parity, not
a speedup** — this model pair sits in the "dead zone" (Pitfall 4) where the draft
forward isn't cheap enough relative to the target to pay for itself.

That raises an obvious question a Mac-only repo can't answer on its own: is
that parity a property of speculative decoding on this hardware, or a
property of *this specific pair, on this specific hardware*? The only way to
tell is to run the same question — does speculation pay for itself, and does
that answer hold as concurrency rises — against a different model pair, a
different (production) engine, and different (data-center) hardware. Hence:
rent a GPU, once, run vLLM's own EAGLE3 support against a real 8B target, and
see whether the shape of the curve matches.

## What I ran

Rented a RunPod A40 (not the A100 originally planned — Lambda Labs card
issues forced a provider switch mid-setup). Target `Llama-3.1-8B-Instruct`,
draft `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`, `num_speculative_tokens=3`,
greedy (`temperature=0`), full GSM8K test split (1319 rows) as the prompt
source, `max_tokens=1024`. Three arms — `eagle3`, `ngram` (vLLM's built-in
prompt-lookup decoding, no draft model), `baseline` (no speculation) — swept
across concurrency ∈ {1, 4, 16, 32, 64}, `guidellm` driving the load against
vLLM's own OpenAI-compatible server. A fourth planned arm, `draft_model`
(small-model-as-draft instead of a trained head), never got past `vllm serve`
startup — see Pitfall 29.

Every arm × concurrency point runs for a fixed 60-second window
(`--constraint kind=max_duration,seconds=60`) rather than to a fixed request
count — necessary because `guidellm run` has no default stopping point at all
against a real (non-synthetic) dataset (Pitfall 27), and I found that out by
watching a concurrency=1 baseline run 8 minutes with no end in sight.

## The result

Real numbers, not estimates — `results/bullet2_vllm_eagle3.json`, generated
by `orchestrate.py --demo-js` from the raw `guidellm` output in
`results/cloud_bench_raw/`:

| concurrency | baseline tok/s | eagle3 tok/s | eagle3 speedup | ngram tok/s | ngram speedup |
|---:|---:|---:|---:|---:|---:|
| 1  | 34.2 | 80.6 | **2.36×** | 51.8 | 1.52× |
| 4  | 121.8 | 300.0 | **2.46×** | 153.4 | 1.26× |
| 16 | 441.4 | 1066.7 | **2.42×** | 562.6 | 1.28× |
| 32 | 829.0 | 1856.7 | **2.24×** | 1041.3 | 1.26× |
| 64 | 1434.3 | 2333.5 | **1.63×** | 1562.4 | 1.09× |

The shape is exactly the one prior art predicts and this repo's own plan
called out before running anything: **the speedup collapses as concurrency
rises.** EAGLE3 goes from 2.36× at concurrency=1 down to 1.63× at
concurrency=64; n-gram from 1.52× down to 1.09× — essentially gone. The
mechanism is the standard one: at low concurrency the GPU is decode-bound and
memory-bandwidth-starved (batch=1 token-by-token generation on an A40 leaves
most of its compute idle), so verifying `k` draft tokens in one forward pass
is nearly free — you're paying for a memory read you'd have paid for anyway.
As concurrency rises, the server is already batching real work across
sequences, so the marginal forward pass speculation buys is no longer close
to free, and the extra verification width competes with other sequences for
the same batched compute.

## What's fundamental vs. what's hardware

The Mac-local finding was: *this pair, on this hardware, is at parity — spec
decoding buys nothing.* The cloud run's finding is not "spec decoding always
wins on real hardware" — it's that **the same collapse-at-concurrency shape
holds on a completely different stack**, and that whether speculation pays
off is a question about *where on the batch-size curve you're operating*,
not a fixed property of the technique. A production server sized to run near
saturation (high concurrency, the common case) sees a much smaller win than a
single-user or low-traffic deployment (concurrency 1–4) — which is also where
this repo's own Mac numbers live (local single-user inference is inherently
low concurrency). The Mac parity result and the A40's 2.36×-at-c1 aren't in
tension; they're two points that make the same curve legible from opposite
ends: a bad draft/target size ratio (Mac) and a saturated server (A40 at
c=64) both erode the same underlying margin, for different reasons.

## Honest caveats

- **Small sample size at low concurrency.** Each 60-second window at
  concurrency=1 completes only 11–29 requests (baseline 11, eagle3 26, ngram
  13); at concurrency=64 it's 457–868. The throughput means are stable (they're
  sums over the window), but the TTFT/TPOT percentile fields are visibly
  noisier at c=1/c=4 than at c≥16 — p99 TTFT swings from ~1.2–1.9s at c=1/c=4
  down to ~90–410ms at c≥16, which is at least partly a sample-size artifact
  of measuring a tail statistic off a dozen points, not purely a
  concurrency effect.
- **One provider, one GPU, one 60-second window per point** — this is a
  sanity-scale cloud run (~$1–2 of A40 time), not a multi-seed reproduction.
  The point was to check the *shape* of the curve against the Mac-local
  finding and prior art, not to publish a tight confidence interval.
- **`draft_model` arm is absent, not zero.** vLLM 0.10.2's V1 engine doesn't
  support that method at all (Pitfall 29) — the table above compares EAGLE3 and
  n-gram against baseline, not all four originally planned arms.
- **A40, not A100.** The GPU actually rented differs from the one named in
  earlier planning notes (payment issues with the original provider forced a
  mid-setup switch) — a different card's memory bandwidth would shift the
  absolute crossover point without changing the qualitative shape.
