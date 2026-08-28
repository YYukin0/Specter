# Specter — a local inference-acceleration engine, built from scratch

Hand-written speculative decoding · KV-cache-correct serving loop · real int4 on
Apple silicon · a fault-injection oracle stack for testing the decoder itself.

本地推理加速引擎：手写投机解码 × KV cache × 真 int4 × 投机解码器的故障注入测试。

> **How do you know your inference optimization is correct? How do you know it's
> actually faster?**

That question is the spine of this repo. Every technique here — speculative
decoding, KV caching, AWQ int4 — is implemented from scratch, tested to
token-level equivalence with its reference, and reported with its failure cases
and null results attached. Speculative decoding on this model pair is at
*parity*, not a speedup; the adaptive-γ controller *loses* to a fixed γ; the
circuit breaker trips once in sixteen runs. Those are in here on purpose. The
value is the verification and measurement discipline, not a claim to a faster
method.

This repo is an engineering study, not a paper. Everything runs locally on a
24 GB M-series Mac at $0 cloud spend. The interesting content is the
**engineering notes** below: each one is "a measurement that fooled me, and how I
caught it."

---

## Engineering notes

Start here. Each is a self-contained "a measurement that fooled me, and how I
caught it." Listed strongest-first; the file prefixes are build order.

1. [Testing a speculative decoder: what output-equivalence checks miss](docs/engineering-notes/06-testing-a-speculative-decoder.md)
   — an oracle lattice (symbolic / structural / batch / real-model), ~20 semantic
   mutation operators, and the finding that a real-model output oracle is *not* a
   superset of the symbolic one — output-equivalence is the weakest test in the
   stack at any model fidelity.
2. [Perplexity is not accuracy — and it mis-ranked two quantizers](docs/engineering-notes/07-perplexity-is-not-accuracy.md)
   — running the self-built AWQ through GSM8K + IFEval: 4-bit costs ~1.4 ppl but
   9.5 points of grade-school math, and perplexity ranks two AWQ implementations
   in the *opposite* order from how they reason.
3. [Getting the KV cache right](docs/engineering-notes/02-getting-the-kv-cache-right.md)
   — the partial-acceptance rollback anchor is `prefix + n_accepted`, not
   `prefix + k + 1`; why the wrong formula passes on high-acceptance prompts.
4. [A flat curve that matched the hypothesis — because the knob was stuck](docs/engineering-notes/01-confirmation-bias-flat-curve.md)
   — an AWQ calibration-size ablation that was silently feeding the quantizer
   the same 512 tokens at every point.
5. [The batch correctness tax](docs/engineering-notes/03-the-batch-correctness-tax.md)
   — per-sequence caches give output-equivalence by construction and a flat
   throughput curve; the realignment cost a padded batch would pay, measured.
6. [The circuit breaker: a real signal, and a stale premise](docs/engineering-notes/04-circuit-breaker-real-signal.md)
   — why a batch-blind cost metric makes "always speculate" unbeatable, and the
   rebuild on a real rolling acceptance rate.
7. [Fake-quant vs real int4](docs/engineering-notes/05-fake-quant-vs-real-int4.md)
   — the number you can report depends on which runtime your backend supports;
   `mlx_lm.gptq` degeneracy as a first-class result.
8. [A fused Metal kernel that moved 2.6× less memory and ran 1.0× as fast](docs/engineering-notes/08-a-fused-metal-kernel-and-the-roofline.md)
   — a hand-written kernel for the accept/reject step vs MLX's op graph: the
   roofline says memory-bound, but a single-threadgroup kernel can't saturate
   bandwidth, and the whole op is ~2% of a target forward. `mx.compile` already
   won.

[**docs/pitfalls.md**](docs/pitfalls.md) — the full trap log (坑1–25), the
build-time ones first.

---

## Where this sits (prior art)

The correctness work draws on two established lines and is careful not to
reinvent either:

- **Batch invariance / numerical determinism** — Thinking Machines,
  [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  (He et al., 2025): batch-invariant RMSNorm / matmul / attention kernels so the
  same input gives the same bits across batch sizes. That is a **kernel-layer**
  property. Oracle O5 here ([note 06](docs/engineering-notes/06-testing-a-speculative-decoder.md))
  is a batch-invariance check at the **loop layer** — it holds by construction
  because the batched decoder shares no ragged tensor — and is not claimed as a
  new idea.
- **Mutation testing** — [cosmic-ray](https://github.com/sixty-north/cosmic-ray)
  and [mutmut](https://github.com/boxed/mutmut) mutate *syntax* (literals,
  operators). `spec_faultlib`'s ~20 operators are *semantic* (crop a KV cache to
  the wrong anchor, draw the bonus token from the draft, freeze the position
  vector) — a hand-written harness because a literal-mutation engine can't
  express those and would drown in equivalent mutants on numerical code. The
  *methodology* — operators → oracle battery → kill matrix (mutation adequacy) —
  is standard.
- **Downstream eval** — the [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  drives GSM8K + IFEval over an OpenAI-compatible endpoint (`local-chat-completions`
  → mlx-lm server), so the from-scratch AWQ model is scored against the same
  baseline on the same config ([note 07](docs/engineering-notes/07-perplexity-is-not-accuracy.md)).
- **Roofline** — Williams et al.'s roofline model (2009) and the
  [MLX custom-kernel API](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
  [Note 08](docs/engineering-notes/08-a-fused-metal-kernel-and-the-roofline.md)
  is a textbook roofline case study on the one op unique to speculative decoding;
  its conclusion — `mx.compile` on a clean op graph beats a single-threadgroup
  hand kernel — is the expected one, and the note is explicit that it's a
  calibration exercise, not an optimisation.

The algorithm-level control-flow bugs this repo hunts (off-by-one cache lengths,
non-contiguous position ramps, swapped distributions) are a **different layer**
from kernel-level float nondeterminism. See
[note 06](docs/engineering-notes/06-testing-a-speculative-decoder.md) for the
split. The batched-decoding correctness framing (keep the batch rectangular or
prove output-equivalence per sequence) follows EQSPEC / *Batch Speculative
Decoding Done Right* (arXiv:2510.22876); this repo takes the per-sequence-cache
side of that fork ([note 03](docs/engineering-notes/03-the-batch-correctness-tax.md)).

---

## What's built

Model pair throughout: draft `Qwen2.5-0.5B-Instruct`, target
`Qwen2.5-1.5B-Instruct`.

| Area | Module | State |
|------|--------|-------|
| Rejection-sampling speculative decoder (greedy + sampling, exact) | `src/rejection_sampling.py`, `src/spec_batch.py` | done |
| KV-cache-correct single-sequence decoder | `src/spec_kv.py` | done |
| Output-equivalent batched decoder | `src/spec_kv_batch.py` | done |
| Continuous-batching serving loop + real-signal circuit breaker | `src/serving_loop.py`, `src/circuit_breaker.py` | done |
| Adaptive-γ controller (GammaTune-style) | `src/gammatune.py` | done — **null result** on this pair (α ≈ 0.79, too little variance) |
| AWQ quantization, from scratch (activation stats → scaling → quantize → ppl) | `src/awq_*.py` | done |
| Real int4 via mlx-lm (AWQ / RTN / GPTQ arms) | `src/verify_p6_2_real_int4.py` | done |
| Downstream eval (GSM8K + IFEval via lm-eval-harness) of self-AWQ vs fp16 vs mlx int4 | `src/build_self_awq_hf.py`, `src/verify_p6_6_downstream_eval.py` | done |
| Fault-injection library (20+ mutation operators) | `src/spec_faultlib.py` | done |
| Oracle stack O1/O3/O4/O5 + O2 (real model) | `src/spec_oracles.py`, `src/verify_p6_5_o2.py` | done |
| Rule-based differential debugger | `src/specdiff.py` | done |
| Fused Metal kernel for the accept/reject step + roofline case study | `src/metal_accept_kernel.py`, `src/verify_p6_7_metal_roofline.py` | done — **negative result**: 2.6× less memory traffic, ~1.0× the speed; `mx.compile` wins |
| Demo: self-contained lab page replaying a recorded real run (+ optional live stdlib HTTP/SSE backend) | `docs/site/index.html`, `src/serve_http.py` | done — open the HTML |

220 tests (`pytest`). Result JSONs for every experiment in `results/`.

---

## Honest headline numbers

On this 0.5B/1.5B pair, on this Mac — not universal claims:

- **KV cache is the big lever.** Target-only decoding: 24.4 tok/s with a KV
  cache, 9.0 without (2.7×). Any speculative decoder that skips the cache is
  competing with itself handicapped.
- **Speculative decoding is at parity here, not a speedup.** `spec_kv` runs at
  0.93–1.0× of KV-cached target-only. The draft forward isn't cheap enough
  relative to the target at batch 1 on Apple silicon — this pair is in the
  "dead zone" (坑4). It is **token-exact** with target-only greedy across
  γ ∈ {1,3,5}, 8 prompts, 3 seeds.
- **Batching this design doesn't add throughput.** Per-sequence caches →
  aggregate tok/s is flat across widths 1–8 (0.96–1.03× of target-only). The
  realignment tax a padded batch *would* pay peaks at width 4 (p90 ≈ 0.68–0.74)
  and collapses by width 8.
- **Real int4 (mlx-lm AWQ):** weights 3.09 → 0.84 GB (**3.7×**), decode
  31 → 104 tok/s (**3.3×**), wikitext-2 ppl **+1.60**. AWQ calibration buys
  **+0.66 ppl** over naive round-to-nearest. `mlx_lm.gptq` produced a degenerate
  model on this architecture (坑21).
- **4-bit costs reasoning, and perplexity doesn't see it.** On GSM8K (identical
  chat/few-shot config, `flexible-extract`): the from-scratch AWQ drops **9.5
  points** (64.8 → 55.3), the mlx-lm int4 model drops **4.0** — while IFEval
  moves ≤ 2.5. Perplexity ranks the two AWQ builds *backwards* vs GSM8K: the
  self-built one is 0.2 ppl **better** and 5.5 points **worse** at math (坑23).
- **Adaptive-γ:** no win on this pair. The acceptance rate is high and stable, so
  there's nothing for a controller to adapt to — confirmed on 3 model pairs.
- **A hand-written Metal kernel for the accept/reject step ties `mx.compile`.**
  The fused kernel moves **2.6× fewer bytes** than the naive MLX op graph and
  runs at **~1.0×** its speed (0.97–1.16× across runs) — a single threadgroup
  can't get past ~19% of the ~84 GB/s bandwidth peak while MLX's multi-kernel
  path reaches ~48%. And the whole step is **~2.3% of one target forward** (zero
  under greedy). Roofline ridge ≈ 33–36 flop/byte; this op sits at ≈ 0.4–1.5,
  memory-bound (坑24).
- **Testing:** against ~20 semantic mutation operators, the output-equivalence
  oracles have a mutation score of ~0.56 (greedy) to 0.75 (sampling) — and
  **0.0 against the KV-cache-management fault class**. The real-model greedy
  oracle (O2) catches only 5 of 9 operators the symbolic oracle (O1) kills and
  misses all three position-index faults: a real model is *not* a superset of the
  fake one. Cache and position bugs are killed only by O4's structural
  invariants. Output-equivalence checks, at any model fidelity, are the weakest
  test in the stack.

---

## Demo

**[`docs/site/index.html`](docs/site/index.html) — open it in a browser.** No
server, no model download, nothing to keep running: a real run of the serving
loop (`src/serving_loop.py`) over four prompts was captured token by token on an
M-series Mac and is embedded in the page, replayed frame for frame. You get the
per-round telemetry the terminal demo prints — γ, accept length, rolling
acceptance rate α, tok/s, concurrency, circuit-breaker state — a mode strip for
the whole run, a throughput trace, and the streamed text, then a second pass
with speculation off. The recorded run: **speculative 25.9 tok/s vs baseline
24.6, 1.05×** — parity, as the rest of this README says.

The folder is self-contained, so it also publishes as-is to GitHub Pages.

To run it live against your own prompts instead of the recording (loads the two
Qwen models, ~5 GB RAM, for as long as you keep it up):

```bash
python -m src.serve_http        # serves the same page at :8137, /generate does real inference
```

The page shows a "run your own prompt" box when it detects that backend.
`src/serve_http.py --capture docs/site/sample_run.json` regenerates the
recording. Terminal version of the same loop:

```bash
python -m demo.live --compare            # or: python -m demo.live --fake   (no download)
```

---

## Layout

```
src/            implementation + one verify_*.py driver per experiment
                serve_http.py — the demo server (stdlib http.server + SSE)
tests/          220 pytest tests (hermetic + model-gated)
results/        one JSON per experiment, committed
docs/
  engineering-notes/   the 8 stories above
  pitfalls.md          坑1–25
  site/                self-contained lab page + the recorded run it replays
notes/          project plan (v9), literature reviews — Chinese, working notes
papers/         reference index (PDFs not vendored; papers/download*.sh)
```

## Running

```bash
python -m pytest -q
```

Experiment drivers are `src/verify_*.py`; the model-backed ones expect the Qwen
weights in the local Hugging Face cache and run offline (`HF_HUB_OFFLINE=1`).
`verify_p6_6_downstream_eval.py` also needs an isolated `.venv-lmeval` with
`lm-eval[api]` (it shells out to it so the main env keeps its pinned versions).
