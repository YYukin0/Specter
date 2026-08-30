# Pitfalls

A running log of the traps this project hit or deliberately steered around. The
numbered list mirrors `notes/project_plan_v9.md` §9.2 — this file is the
English, standalone version, ordered so the ones found *while building* come
first because those are the ones worth reading.

Each entry: what it was, why it was easy to get wrong, what was done about it.

---

## Found while building (Pitfalls 13–29; 13–21 added 2026-08-28, 22–26 added 2026-08-29, 27–29 added 2026-08-30)

### Pitfall 29 — vLLM's V1 engine flatly refuses draft-model speculative decoding
**Where:** Bullet 2 (Pillar 7), full arm × concurrency matrix on the rented A40.

The plan's fourth arm, `draft_model` (a small same-family model as the draft
instead of a lookahead/eagle head), crashed the `vllm serve` process before it
ever answered `/health`: `NotImplementedError: Speculative decoding with
draft model is not supported yet. Please consider using other speculative
decoding methods such as ngram, medusa, eagle, or deepseek_mtp.` — raised
from `_is_v1_supported_oracle` in `vllm.engine.arg_utils`, on the very first
line of engine construction. Not a config mistake to fix: `vllm==0.10.2`'s V1
engine (the only engine, V0 was removed 2025-11-12) simply doesn't implement
this method yet.

**Compounding bug:** `orchestrate.run_matrix`'s default arm list fell back to
`[s.name for s in config.arm_specs()]` — all 4 arms including `draft_model` —
rather than the curated `config.ARM_NAMES`. That's exactly what let the
unsupported arm into a real run: `orchestrate.py --execute` with no `--arms`
flag ran it by default. The crash left `vllm serve` dead and the orchestrator
stuck in `wait_for_health`'s 5-minute timeout, waiting on a server that would
never come up — had to `kill -9` it manually rather than wait it out.

**Fix:** `config.ARM_NAMES` now excludes `draft_model` (3 arms:
eagle3/ngram/baseline); `arm_specs()`/`arm_spec_by_name()` still define it for
the record. `run_matrix`'s default reads `ARM_NAMES`, not `arm_specs()`.

**Lesson:** a "supported speculative-decoding methods" list buried in a
`NotImplementedError` string is easy to miss until you actually try to start
the server — `vllm serve --help` doesn't enumerate per-method support, only
the `--speculative-config` schema shape. Test the arm that isn't in the
method's own advertised list before building a matrix around it.

---

### Pitfall 28 — a placeholder GuideLLM output schema guess never matched, silently
**Where:** Bullet 2 (Pillar 7), `orchestrate.normalize_guidellm_result`.

Written before any real `guidellm run` output existed, this function dug for
fields at `raw["metrics"]["output_tokens_per_second"]["mean"]` with a flat-shape
fallback. Neither path exists in the real `guidellm==0.7.3` output: the true
shape is `raw = {"metadata", "config", "benchmarks": [...]}`, and each
`benchmarks[0]["metrics"][<name>]` is a per-completion-status breakdown
(`{"successful": {...distribution stats...}, "errored": {...}, ...}`). The
first real sanity-check run completed cleanly end-to-end — server up,
GuideLLM ran, JSON written — and `normalize_guidellm_result` returned every
field as `null` without raising anything. `compute_speedup` would have divided
by `None`, not by zero; the bug was caught by eyeballing the checkpoint JSON,
not by a crash.

**Second layer of the same bug:** even after pointing at
`benchmarks[0]["metrics"][name]["successful"]`, the percentile fields
(`ttft_p99_ms`, `tpot_p99_ms`) still came back `null` — `p99` lives one level
deeper, under `successful["percentiles"]["p99"]`, not as a top-level key on
the stats object itself alongside `mean`/`median`/`min`/`max`.

**Fix:** rewrote the function against the real output JSON on the rented box
(`results/cloud_bench_raw/guidellm_baseline_c1.json`), with a `stat()` helper
that routes percentile-shaped names (`p99`, `p50`, ...) through the nested
`percentiles` dict and everything else (`mean`, `median`, ...) as a direct key.

**Lesson:** a schema-mapping function with no real example to check against
degrades silently to all-`None` rather than erroring — there's no `KeyError`
to catch when every `.get()` has a default. The only way to know it's wrong is
to look at what it actually produced, once, against real data.

---

### Pitfall 27 — GuideLLM has no default stopping point against a real dataset
**Where:** Bullet 2 (Pillar 7), `orchestrate.guidellm_cmd`.

The first real sanity-check run — `baseline` arm, concurrency=1, against the
full 1319-row GSM8K test split — was still running 8+ minutes in with no end
in sight: ~34 tok/s generation, up to 1024 output tokens per request,
concurrency 1 means fully sequential requests. `guidellm run` has no implicit
request-count or time cap; left alone it walks the entire dataset, which at
this rate would have taken on the order of hours for one arm × concurrency
point out of 15. Had to kill the run and a leftover `VLLM::EngineCore` process
(still holding 41 GB of GPU memory after the parent `vllm serve` was killed)
by PID to actually free the GPU.

**Fix:** `--constraint kind=max_duration,seconds=60` (field name
`MaxDurationConstraintArgs.seconds`, found by importing
`guidellm.scheduler.constraints` directly and reading the pydantic model,
cross-checked against `guidellm run --help`'s `--constraint kind=[max_errors|
max_error_rate|max_global_error_rate|max_duration|max_requests|
over_saturation],...`) bounds every arm × concurrency point to the same
wall-clock budget instead of a request count — matches the 60s/point figure
the execution plan had written down before any of this was verified.

**Lesson:** a benchmarking tool measuring "requests per second" doesn't
imply it will ever decide it has *enough* requests — check for a stopping
condition before launching against a real (as opposed to a handful of
synthetic) dataset, on a fixed-cost rented resource, at concurrency=1 where
each request is fully serial.

---

### Pitfall 18 — the partial-acceptance rollback formula in the plan was wrong
**Where:** P6.0, single-sequence KV-cache speculative decoding (`src/spec_kv.py`).

The plan said: on a partial acceptance, crop the target KV cache to
`prefix + k + 1` and the draft cache to `prefix + k` (different lengths, the
extra slot on the target side "for the bonus / resampled token").

That is wrong. In one round the target forward is fed
`[pending_target_token] + [γ draft tokens]` and produces `γ+1` rows of logits.
The token at position `k` — whether it's an accepted draft token, the
resampled token on a rejection, or the bonus token on a full acceptance — is
the **output** of that forward. It was never an **input** to any forward, so
its KV does not exist. The correct move is: **crop both caches to
`prefix + n_accepted`** (`prefix + γ` on a full acceptance).

**Why it's easy to get wrong:** on a *full-acceptance* round `prefix + k + 1`
happens to equal the current sequence length, so a crop helper that early-returns
when `target_len >= current` silently does nothing and the behaviour looks
correct. Only a round with an actual rejection exposes it.

**Fix:** implement `prefix + n_accepted`; add a `_crop_to(cache, target_len)`
wrapper that only accepts `target_len <= current` and pins the `DynamicCache.crop`
negative-offset semantics; add an always-on test invariant
`cache.get_seq_length() == len(committed) - 1` after every round; add a
rejection-heavy stress phase (`phase=2.4`) so non-full-acceptance rounds are
actually exercised. Cross-checked against HuggingFace
`generation/utils.py::_speculative_sampling` (`crop(-(candidate_length - n_matches))`),
which is the same `prefix + n_matches`.

**Lesson:** the anchor for a rollback length is "how many tokens were fed
through a forward", not "how many tokens this round touched".

---

### Pitfall 19 — the batch correctness/throughput trade-off is real; measure the tax you don't pay
**Where:** P6.1, output-equivalent batched speculative decoding (`src/spec_kv_batch.py`).

Every naive way to batch speculative decoding (masking, rollback, dynamic
padding) breaks output equivalence: different sequences accept different numbers
of draft tokens per round, so position ids / attention masks / KV-cache lengths
drift apart across rounds (EQSPEC, arXiv:2510.22876).

The choice here was **not to fix the masking path but to sidestep it**: each
sequence gets its own KV cache, and a "batch round" runs the already-verified
single-sequence step once per sequence. No shared ragged tensor → nothing drifts
→ output equivalence holds **by construction** (a test pins
`batched == N × single-sequence`, bit-exact, greedy and sampling).

**Cost:** no kernel-level ragged-verify batching, so throughput is nearly flat
in concurrency (`speedup_vs_width1` 0.94–1.03). *That flatness is the finding.*

**Measure the tax that wasn't paid:** the realignment overhead an EQSPEC-style
padded-batch path *would* pay is computed analytically against the rectangular
counterfactual: `1 - Σ work / (n · max work)`. Mean rises to 0.073 (short) /
0.096 (long) at width 4, p90 0.68–0.74; it drops back to 0.007–0.02 at width 8
(work is more uniform when every sequence runs every round). The tax peaks at
*medium* width.

**Lesson:** an implementation that can prove output equivalence gives up the
batch speedup; one that gets the speedup pays a realignment tax. Report the tax
as a first-class measurement, don't pretend batch speedup is free.

---

### Pitfall 20 — a circuit breaker on a synthetic signal can't be evaluated honestly
**Where:** P5.3 → P6.1, the batch-aware circuit breaker.

P5.3's cost model was `total_emitted / total_cost_units` with a speculative round
costing `c + γ` and a degraded step costing `c`. **No term in that formula
depends on batch size.** So "always speculate" is a structural upper bound the
breaker can't beat: degrading to plain target decoding does strictly less useful
work per unit compute, and there's no "draft forward gets expensive under a
saturated accelerator at high batch" term to compensate. Measured: always-spec
0.280 ± 0.010 vs breaker 0.247 ± 0.008 — the primary metric says "breaker is
useless", because the batch signal was synthetic and never fed back into the
acceptance rate α.

**Fix in P6.1:** the breaker trips on a **real** rolling α (windowed mean of the
last `alpha_window` acceptance decisions across all sequences) `< alpha_floor`,
optionally also on a target-only latency probe showing speculative rounds are
wall-clock slower. `len(active)` is an *input*, never the rule. While degraded,
force a speculative probe every `reprobe_every` rounds so α stays observable
(Pitfall 11). Over 16 runs the breaker tripped exactly once (`long / width 2 /
breaker on`) and **never** because of batch size (width 8 never degrades).

**Lesson:** a circuit breaker only earns its keep on a workload where α actually
drops. Under a healthy α it should be a near-no-op — "never degrades" is not a
bug. Honest validation needs a pair / distribution that genuinely drives α down.

---

### Pitfall 21 — `mlx_lm.gptq` produced a degenerate model; sentinel-check every third-party artifact
**Where:** P6.2, real int4 via mlx-lm.

`mlx_lm.gptq` (mlx-lm 0.31.3), Qwen2.5-1.5B-Instruct, bits=4 group-size=128:
the output model emits a constant `!` and its wikitext-2 forward NLL is `nan`.
First guess was calibration starvation (the first run gave GPTQ only 2 Hessian
batches from `--num-samples 16`), so it was re-run at `--num-samples 64
--sequence-length 512` — **still degenerate**. `mlx_lm.awq` and
`mlx_lm.convert -q` RTN on the identical model/config are both fine.

Not bisected further (group size? a bad fallback layer? an mlx-lm bug for the
Qwen2 architecture?) — compute budget, and AWQ was already the primary real-int4
arm.

**Fix:** the verify script sanitizes a non-finite perplexity to `null` +
`degenerate_forward: true`, flags the arm, records `None` for its delta, prints
`GPTQ DEGENERATE` in the headline, and keeps the JSON strict (no `NaN` token).
The result file's `quant_config.gptq` documents the full repro.

**Lesson:** when you borrow a third-party quantizer as a comparison arm, the
output model must pass a sentinel — "generate one sentence, is it words?" plus
"one forward, is the NLL finite?" — before you trust it. Don't just check that
the tool finished and the weights file exists. And keep the degenerate arm in
the results: "this tool doesn't work on this architecture" is useful information.

---

### Pitfall 22 — GSM8K `strict-match` is a format check, not an arithmetic check, when the model is a chat model
**Where:** P6.6 (Pillar 7 Bullet 3), downstream eval of AWQ.

lm-eval's GSM8K task reports two numbers: `strict-match` (the answer must appear
as `#### <number>` at a fixed position) and `flexible-extract` (last number in
the response). With a chat template, Qwen2.5-1.5B produces a conversational CoT
that ends "…so the total is **72**." — it rarely emits the `####` anchor. The
**fp16 baseline itself** scores 0.378 strict vs 0.648 flexible. So the
strict-match *deltas* between quantized and baseline (−22, −17 points) are mostly
measuring how often each model happens to hit the format, not how often it does
the math.

**Fix:** report `flexible-extract` as the GSM8K metric; treat `strict-match` as
noise on this stack. Keep both in the result JSON, lead with flexible in the
write-up. (Cousin of lm-eval issue #1841 — chat templates silently move scores.)

**Lesson:** know what each metric variant actually rewards before you quote a
delta on it. A metric the baseline already fails for reasons unrelated to your
change is not a measurement of your change.

---

### Pitfall 23 — perplexity mis-ranked two 4-bit AWQ implementations for reasoning
**Where:** P6.6 (Pillar 7 Bullet 3).

On the identical wikitext-2 harness, the self-built AWQ scores **+1.39 ppl** vs
fp16 and the `mlx_lm.awq` int4 model scores **+1.60** — so perplexity says the
self-built one is the better quantizer. On GSM8K flexible-extract the order
**reverses**: self-built loses **9.5 points**, `mlx_lm.awq` loses **4.0**. A
0.2-ppl "win" on prose corresponds to a 5.5-point *deficit* on grade-school
math. IFEval barely moves for either (−2.5 / −1.0 pt), so the damage is specific
to multi-step reasoning, where per-weight rounding error compounds across CoT
steps.

The two arms differ in scale/clip search (`mlx_lm.awq` does the full AWQ
weight-clip search; the self-built path does the scale search only) and in
calibration and fp16-fallback policy — differences that barely register on
perplexity but move GSM8K by ~5 points.

**Fix / practice:** perplexity is a screening metric, not an acceptance metric.
Before trusting a quantized model, run it on at least one task that needs a
correct multi-step output, with the eval config (chat template, few-shot format,
decoding params) held **identical** to the baseline.

**Lesson:** ppl and downstream accuracy are not the same measurement and here they
don't even agree on sign. "AWQ 4-bit g128" is not one number — the search and
calibration details decide whether the model can still reason.

---

### Pitfall 24 — a fused Metal kernel NaN'd only when the vocab was smaller than the threadgroup
**Where:** P6.7 (Pillar 7 optional), `src/metal_accept_kernel.py`.

The fused accept/reject kernel does a one-pass online softmax per row: each
thread keeps a running max `m` and running sum-of-exp `s` over its stride of V,
then the threadgroup merges the `(m, s)` pairs with
`mM = max(mA, mB); sM = sA·exp(mA−mM) + sB·exp(mB−mM)`. With a 1024-thread
threadgroup and the real vocab V = 151936, every thread does work and it's fine.
With V < 1024 (the smoke config, and the V = 256 unit test), the surplus threads
never enter the sweep and carry the identity `(m, s) = (−∞, 0)`. Two of them
merging gives `mM = −∞` and `exp(−∞ − (−∞)) = exp(NaN) = NaN`, which then
propagates through the whole row.

**Fix:** guard the merge — `sM = (mM > −INFINITY) ? (…) : 0`. The identity
element stays `(−∞, 0)`.

**Lesson:** this is exactly the bug a proportionally-shrunk smoke test is
supposed to catch and a "just use realistic sizes" test would miss — the failure
mode lives at `n_threads > vocab`, a boundary the full-size run never crosses. The
kernel tests use V = 256 on purpose.

---

### Pitfall 25 — an SSE stream over HTTP/1.1 keep-alive hangs the client forever
**Where:** P6.8 (Pillar 7), `src/serve_http.py`.

The first smoke test of the `/generate` endpoint never returned — a 2-minute
timeout with the server process still alive. `BaseHTTPRequestHandler` with
`protocol_version = "HTTP/1.1"` keeps the connection alive by default. The
response had no `Content-Length` (it can't — the stream length isn't known up
front) and no `Transfer-Encoding: chunked`, so after the last
`event: done\n\n` the client sat waiting for either more bytes, a length it
would never get, or a close that never came.

**Fix:** send `Connection: close` and set `self.close_connection = True` on the
handler, so the socket EOF *is* the end-of-stream signal. The browser's
`fetch()` reader and `curl -N` both terminate cleanly on EOF. (The alternative —
implement chunked encoding by hand — buys nothing here; one generation owns the
connection anyway.)

**Lesson:** "no `Content-Length`" only means "read until EOF" if the server
actually closes. Under keep-alive it means "hang". Test the client's
end-of-stream path, not just that bytes arrive.

---

### Pitfall 26 — a circuit-breaker demo scenario that tripped once and never came back
**Where:** P6.8 polish (Pillar 7), the "Breaker trips" replay scenario in
`src/serve_http.py`.

To make the mode strip visibly cycle spec → degraded → probe → spec on the
demo page, the first attempt pushed `alpha_floor` far above the default (0.92
vs 0.5) on the theory that "higher floor = trips more reliably." It did trip —
immediately, on round 2 — but then never recovered: the rolling α for this
draft/target pair on this prompt oscillates in a band roughly 0.5–0.75, so a
0.92 floor sat entirely above the band the model could ever reach. Every
periodic re-probe (Pitfall 11's mechanism) measured α, found it still short of 0.92,
and went straight back to degraded. The strip showed exactly one green bar
followed by a wall of red with amber re-probe ticks — not the trip-and-recover
cycle the scenario was supposed to demonstrate.

**Why it's easy to get wrong:** "trip more" and "trip and recover" sound like
the same knob turned further in the same direction, but they're opposite
requirements. Reliably tripping wants the floor *above* the band; reliably
*recovering* wants the floor *inside* the band, so some rounds clear it and
some don't.

**Fix:** measure the pair's actual rolling-α range on the target prompt first
(printed the per-round sequence for a scratch run), then set `alpha_floor`
just inside that band (0.6) rather than safely above it. Confirmed the
resulting capture actually cycles (spec 53 / degraded 12 / probe 3 rounds,
several full spec→degraded→probe→spec loops) before committing the recording.

---

### Pitfall 16 — an AWQ calibration-size ablation with a stuck knob
**Where:** P2.3, AWQ calibration-set-size ablation.

The self-built AWQ path captures all 196 target Linear layers' input activations
in a single forward, so it needs a `max_tokens_per_layer` cap to avoid OOM. That
cap was hard-wired to 512, and each calibration sequence was also truncated to
512 tokens. So the *first* 512-token wikitext row filled every layer's pool and
capture stopped immediately. Result: `n_calib ∈ {8, 16, 32, 64, 128}` all fed
the **byte-identical** "first 512 tokens" pool — quantization output and
perplexity were bit-identical across the whole sweep (seed 0: 13.6015 for
n_calib = 4/8/16/32/64).

**Why it was nearly missed:** the flat curve *matched the AWQ paper's claim*
that a small calibration set suffices. Textbook confirmation bias — the review
question should have been "did the `n_calib` knob actually move?", not "is the
curve flat?".

**Fix:** expose `max_tokens_per_layer` / `max_seq_len` (default still 512, so
P2.2 is byte-for-byte unchanged); in P2.3 truncate rows to 64 tokens and set the
cap to `n_calib × 64` so the pool grows linearly; record
`captured_tokens_per_layer` per point plus a `capture_knob_actually_moved`
boolean that auto-annotates the verdict string when `max/min ≤ 1.5×`; drop
`n_calib = 128` (would OOM a 24 GB machine).

---

### Pitfall 17 — the "cached" code corpus was only a README
**Where:** P2.2, cross-distribution AWQ.

Notes said `codeparrot/codeparrot-clean-valid` and `allenai/c4` were cached
locally; in fact only their README snapshots were, with no data shards, so
`HF_HUB_OFFLINE=1` couldn't load them. The "code" distribution fell back to
`google-research-datasets/mbpp`'s `code` field (~176 k chars).

**Why it distorts:** mbpp solutions are 3–8 line, highly templated functions —
far narrower than real code. As a calibration set they *inflate* the
cross-distribution gap (measured +0.56 ppl for calib=code→eval=NL vs
calib=NL→eval=NL, which looks like "AWQ is unstable across distributions" but is
partly the narrow corpus); as an eval set they give an unrealistically low ppl
(fp16 baseline 3.1).

**Fix:** the result file's `code_corpus_note` states the mbpp substitution, the
fp16 baseline number, and that the gap includes a narrow-corpus contribution;
GPTQ arm and real code corpora deferred to a later stage.

---

### Pitfalls 13, 14, 15 — the earlier implementation-hit traps

- **Pitfall 13 (M2, P1.2):** greedy speculative decoding is only token-exact with
  greedy target-only if the tie-breaking and the "compare argmax, not sampled
  token" convention match exactly on both paths; a mismatch shows up as a slow
  drift, not a crash.
- **Pitfall 14 (M4, P5.0/P5.1):** the GammaTune null result — the main model pair sits
  at α ≈ 0.79 with too little variance for an adaptive-γ controller to help;
  confirmed on 3 model pairs (see the Pitfall 9 addendum), so the null is robust, not an
  artifact of one pair.
- **Pitfall 15 (M4, P5.3):** see Pitfall 20 — the synthetic batch signal made "always
  speculate" an unbeatable upper bound for the breaker's cost metric.

---

## Anticipated from prior art (Pitfalls 1–12)

These came out of the literature review *before* implementation and shaped the
design; most were designed around rather than hit.

| # | Trap | Guard |
|---|------|-------|
| Pitfall 1 | Tokenizer / vocab mismatch silently zeros the acceptance rate (even within a family: Qwen2 1.5B vocab 151936 vs 72B 152064). | P1.0 asserts vocab identity. |
| Pitfall 2 | Bonus-token sampled from the *draft* distribution (a real DSD bug) — violates correctness, doesn't crash. | Unit test pins which model's logits the bonus token comes from. |
| Pitfall 3 | `batch > 1` ragged-tensor desync — sequences accept different token counts, so position ids / masks / cache lengths go ragged. | Pillar 4 maintains per-sequence state by hand; see Pitfall 19. |
| Pitfall 4 | Draft/target "dead zone" — a size gap under ~2–3× can be slower than no speculation. AdaEDL reports static SPD 16% *slower* than autoregressive for one bad pair, +43% *faster* once adaptive early-stop is added. | Diagnostic: α is fine but wall-clock regresses → check the draft's own latency share first. |
| Pitfall 5 | Calibration-distribution mismatch overfits GPTQ badly, AWQ less so (cross-distribution: AWQ +0.5–0.6, GPTQ +2.3–4.9). | P2.2 reproduces the matrix. |
| Pitfall 9 | GammaTune degrades under adversarial / highly non-stationary workloads and helps little when draft/target already agree. | P5.1 non-stationary test; the Pitfall 9 addendum confirms the low-variance regime can't be escaped within the Qwen2.5-Instruct family. |
| Pitfall 10 | Optimal γ shifts with target quantization (SpecKV: FP16 γ=2 → INT8 γ=8 under BitsAndBytes, a 4× shift) — quantization and adaptive control are not independent. AWQ's per-channel scaling may shift it *less*; that would be a finding, not an error. | P5.2 reuses the Pillar 2 quant models + a BnB NF4 same-source control arm. |
| Pitfall 11 | DSD-style breakers "stop collecting data when disabled" and can't re-enable; BanditSpec ignores KV-cache rebuild cost. | P5.3/P6.1 breaker has periodic re-probing + measured switch cost (Nightjar's 17.87–102 ms is the sanity band). |
| Pitfall 12 | BanditSpec's K-armed frame doesn't treat batch size as context and its regret bound ignores switch cost. | Cited for comparison only; not reimplemented. |
