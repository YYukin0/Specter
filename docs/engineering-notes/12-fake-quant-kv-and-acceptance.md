# Fake-quantizing the target KV cache, and what it does to acceptance

**Task:** Pillar 8 (支柱8) Track E — quantify how a lossy *target* KV cache moves
the speculative acceptance rate `alpha`, with the draft left in fp16
(independent-draft speculation, the opposite of QuantSpec-style
self-speculation). Measurement, no pass/fail threshold.
**Pitfalls:** 36, 37.
**Code:** `src/kv_fakequant.py`, `src/longctx_prompts.py`,
`src/verify_p7_3_kv_quant.py`, `src/spec_kv.py` (`make_draft_cache` param),
`tests/test_kv_fakequant.py`, `results/p7_3_kv_quant.json`.

## Why hand-rolled

`optimum-quanto`, `hqq`, and `optimum` are all absent from this environment and
none has a Python 3.14 wheel — the same dependency gap note 05 hit for real
int4. So `FakeQuantKVCache` is a *simulation*: it subclasses
`transformers.DynamicCache` and, after every `update()`, round-trips KV
positions older than a full-precision "residual" window through per-channel
symmetric int-N quant → dequant, storing the result back as fp16. It packs
nothing and saves no memory. What it reproduces faithfully is the *numerical
error* an int-N KV would carry into attention — which is the only thing that can
move `alpha`.

This mirrors note 05's caveat about fake-quant vs a real packed int4 build: the
error model is right, the systems properties (memory, bandwidth, a real
dequant kernel's rounding) are not. A real int4 KV would also differ in group
layout and could be marginally kinder or harsher than per-channel-over-sequence
here.

## The residual window keeps rollback away from quantized data

`residual_len = 64` positions stay fp16. Speculative rollback is `crop(-g)` with
`g <= gamma <= 8`, comfortably inside that window, so `crop()` never lands on a
quantized position and there is no "requantize a partially-rolled-back block"
hazard (Pitfall 37). `crop()` is therefore inherited unchanged.
`test_kv_fakequant.py::test_crop_stays_inside_the_fp_residual_window` pins it.

One fidelity wrinkle (Pitfall 36 territory): `_fake_quant` recomputes its
per-channel scale over the *current* `[:cut]` slice on every update, so as the
sequence grows the already-frozen region is re-quantized against a slightly
drifting `amax`. The extra error per step is tiny (the scale is dominated by the
same outliers) but it is not zero — a production cache would freeze a block's
scale once and never touch it again.

## Arms and metrics

`verify_p7_3_kv_quant.py`, real `Qwen2.5-0.5B / 1.5B` on MPS fp16, eight
~800-token prompts (`longctx_prompts.py`: neutral filler + a real question),
`gamma = 3`, 48 new tokens, greedy. Five arms, baseline `(16, "kv")`:

| arm | quantizes | `alpha` | `alpha_delta` | accept_len | KL mean | KL p90 | quality | tok/s |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `(16, kv)` baseline | nothing (passthrough) | 0.802 | 0.000 | 1.97 | 0.0000 | 0.0000 | 1.00 | 14.7 |
| `(8, kv)`  | K + V | 0.802 | 0.000 | 1.97 | 0.0002 | 0.0004 | 0.99 | 14.4 |
| `(4, kv)`  | K + V | 0.411 | **−0.391** | 0.76 | 10.34 | 14.44 | 0.00 | 8.6 |
| `(8, k_only)` | K only | 0.802 | 0.000 | 1.97 | 0.0001 | 0.0003 | 1.00 | 14.3 |
| `(4, k_only)` | K only | 0.329 | **−0.473** | 0.71 | 10.51 | 14.56 | 0.00 | 6.3 |

- **`alpha` / `alpha_delta`** — mean speculative acceptance rate, and its shift
  from the fp16 baseline.
- **KL mean/p90** — over the first 32 decode positions, KL of the target's
  next-token distribution *with* fake-quant against the fp16 target,
  teacher-forced on the fp16 tokens so the two are compared position-for-position.
- **exact-prefix quality** — greedy-decode the fake-quant target alone; how many
  leading tokens match the fp16 target's greedy output before the first
  divergence.

## Direction

The result is a cliff, not a slope. **8-bit** fake-quant on the target KV is
free: `alpha` is unchanged to four decimals, the per-position KL against the
fp16 target is ~2e-4, greedy output is identical for 7 of 8 prompts, and tok/s
drops <3%. **4-bit** collapses it: `alpha` falls from 0.80 to ~0.35–0.41,
`accept_len` drops below 1.0 (speculation is now *net-negative* — the draft
block is rejected more often than not and the extra draft forwards aren't paid
back), per-position KL jumps to ~10 nats, and exact-prefix quality is 0.00 —
the fake-quant target diverges from the fp16 target on the *very first* decoded
token. tok/s roughly halves, from both effects at once: speculation stops
paying, and `_fake_quant` adds a quant/dequant pass over the frozen KV every
step.

The literature direction for *independent-draft* speculation is unambiguous: a
lossy target KV widens the total-variation distance between the draft and target
next-token distributions, and since acceptance is exactly `1 - TV`, `alpha`
falls. QuantSpec and similar self-speculation results — where quantizing the
shared cache *helps* because draft and target degrade together — do **not**
transfer here: the draft is a separate fp16 model and only the target is
degraded, so any target error is pure disagreement.

## K-only vs K+V

KIVI / Kitty report keys as substantially more sensitive to low-bit
quantization than values — key outliers concentrate in a few channels, value
distributions are flatter. The arms here agree qualitatively. At 8-bit both
`kv` and `k_only` are null (as is, by implication, a v-only quant). At 4-bit,
**keeping V in fp16 buys back essentially nothing**: `(4, k_only)` lands at
`alpha = 0.329` vs `(4, kv)`'s 0.411, i.e. slightly *worse*, with near-identical
KL (10.5 vs 10.3). The damage is in the keys; whether V is also quantized is in
the noise of 8 greedy prompts. The nominal inversion (fewer tensors quantized →
lower `alpha`) is not a real effect at this sample size — the honest statement
is "4-bit keys alone already collapse acceptance, and V quantization on top is
not measurably separable."

## Honest boundaries

- **Not a real int-N cache.** No packing, no memory saving, no bandwidth change
  — only the arithmetic error. Systems claims are out of scope.
- **Per-channel-over-sequence, scale recomputed each step.** A shipping cache
  freezes block scales; this one drifts them slightly (above).
- **Single machine, greedy, 48 new tokens.** `alpha` is a mean over eight
  prompts; no temperature sweep, no batch.
- **Long context is concatenated filler**, not natural long-form text, so the
  KV-length regime is right but the token statistics are stylized.
- **Draft stays fp16.** This isolates target-KV error; it says nothing about
  quantizing both, which is a different (and per the self-speculation
  literature, potentially opposite) story.
