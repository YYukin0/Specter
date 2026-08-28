# Specter — a local inference-acceleration engine, built from scratch

Hand-written speculative decoding · KV-cache-correct serving loop · real int4 on
Apple silicon · a fault-injection oracle stack for testing the decoder itself.

本地推理加速引擎：手写投机解码 × KV cache × 真 int4 × 投机解码器的故障注入测试。

This repo is an engineering study, not a paper. Everything runs locally on a
24 GB M-series Mac at $0 cloud spend. The interesting content is the
**engineering notes** below: each one is a bug or a design fork that was worth
writing down, with the numbers that settled it.

---

## Engineering notes

Start here. Each is a self-contained story.

1. [A flat curve that matched the hypothesis — because the knob was stuck](docs/engineering-notes/01-confirmation-bias-flat-curve.md)
   — an AWQ calibration-size ablation that was silently feeding the quantizer
   the same 512 tokens at every point.
2. [Getting the KV cache right](docs/engineering-notes/02-getting-the-kv-cache-right.md)
   — the partial-acceptance rollback anchor is `prefix + n_accepted`, not
   `prefix + k + 1`; why the wrong formula passes on high-acceptance prompts.
3. [The batch correctness tax](docs/engineering-notes/03-the-batch-correctness-tax.md)
   — per-sequence caches give output-equivalence by construction and a flat
   throughput curve; the realignment cost a padded batch would pay, measured.
4. [The circuit breaker: a real signal, and a stale premise](docs/engineering-notes/04-circuit-breaker-real-signal.md)
   — why a batch-blind cost metric makes "always speculate" unbeatable, and the
   rebuild on a real rolling acceptance rate.
5. [Fake-quant vs real int4](docs/engineering-notes/05-fake-quant-vs-real-int4.md)
   — the number you can report depends on which runtime your backend supports;
   `mlx_lm.gptq` degeneracy as a first-class result.
6. [Testing a speculative decoder: what output-equivalence checks miss](docs/engineering-notes/06-testing-a-speculative-decoder.md)
   — an oracle lattice (symbolic / structural / batch / real-model) and the
   finding that a real-model output oracle is *not* a superset of the symbolic
   one.

[**docs/pitfalls.md**](docs/pitfalls.md) — the full trap log (坑1–21), the
build-time ones first.

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
| Fault-injection library (20+ mutation operators) | `src/spec_faultlib.py` | done |
| Oracle stack O1/O3/O4/O5 + O2 (real model) | `src/spec_oracles.py`, `src/verify_p6_5_o2.py` | done |
| Rule-based differential debugger | `src/specdiff.py` | done |

195 tests (`pytest`). Result JSONs for every experiment in `results/`.

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
- **Adaptive-γ:** no win on this pair. The acceptance rate is high and stable, so
  there's nothing for a controller to adapt to — confirmed on 3 model pairs.
- **Testing:** the real-model greedy oracle (O2) catches 5 of 9 mutation
  operators that the symbolic oracle (O1) kills; it misses all three
  position-index faults, which need the structural oracle O4. Output-equivalence
  checks — at any model fidelity — are the weakest test in the stack.

---

## Layout

```
src/            implementation + one verify_*.py driver per experiment
tests/          195 pytest tests (hermetic + model-gated)
results/        one JSON per experiment, committed
docs/
  engineering-notes/   the 6 stories above
  pitfalls.md          坑1–21
notes/          project plan (v9), literature reviews — Chinese, working notes
papers/         reference index (PDFs not vendored; papers/download*.sh)
```

## Running

```bash
python -m pytest -q
```

Experiment drivers are `src/verify_*.py`; the model-backed ones expect the Qwen
weights in the local Hugging Face cache and run offline (`HF_HUB_OFFLINE=1`).
