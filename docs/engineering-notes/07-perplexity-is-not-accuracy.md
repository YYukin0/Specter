# Perplexity is not accuracy — and it mis-ranked two quantizers

**Task:** P6.6 (支柱7 Bullet 3) — run the self-built AWQ model through a real
downstream benchmark (GSM8K + IFEval) instead of stopping at perplexity.
**Pitfalls:** 坑22, 坑23; Bullet 3 pitfalls 1–5.

## Why this step exists

支柱2 measured 4-bit AWQ the cheap way: wikitext-2 perplexity. The self-built
fake-quant path costs about **+1.2–1.4 ppl** on the 1.5B target; the mlx-lm
production int4 model costs **+1.6 ppl** (P6.2). Both look like small, similar
hits. The obvious question the ppl number can't answer: does the model still
*reason*?

So P6.6 stands up an OpenAI-compatible endpoint (mlx-lm server) in front of each
model and drives the **EleutherAI lm-evaluation-harness** (`local-chat-completions`)
over it — GSM8K (5-shot CoT, grade-school arithmetic) and IFEval (0-shot,
instruction-following). One eval config for every arm, differing **only in
weights**: greedy, `max_gen_toks=768`, `--apply_chat_template`,
`--fewshot_as_multiturn`, no system prompt, `--limit 400`, seed 0.

Three arms:

| arm | what it is |
|-----|------------|
| `fp16` | Qwen2.5-1.5B-Instruct, unquantized |
| `self_awq` | the from-scratch AWQ (activation stats → per-channel scale search → 4-bit fake-quant), served at fp16 |
| `mlx_awq_int4` | `mlx_lm.awq` real int4, the P6.2 arm |

## The result

All three arms on the **same** wikitext-2 ppl harness (P6.2 recipe, shared token
array — no cross-harness caveat this time), and the same 400-example GSM8K /
IFEval slice:

| arm | wikitext-2 ppl | Δppl | GSM8K flex-extract | Δ | IFEval prompt-strict | Δ |
|-----|---------------:|-----:|-------------------:|--:|---------------------:|--:|
| fp16          | 11.54 | —      | 0.648 | —      | 0.418 | —      |
| **self_awq**  | **12.94** | **+1.39** | **0.553** | **−0.095** | 0.393 | −0.025 |
| **mlx_awq_int4** | **13.14** | **+1.60** | **0.608** | **−0.040** | 0.408 | −0.010 |

Two things fall out, and the second one is the point of the note.

### 1. Reasoning degrades much more than instruction-following

For `self_awq`, 4-bit costs **9.5 points of GSM8K** (−14.7% relative) but only
**2.5 points of IFEval** prompt-level strict (−6% relative). For `mlx_awq_int4`
it's 4.0 vs 1.0. Multi-step arithmetic is where per-weight rounding error
compounds — each CoT step multiplies the drift forward. Following a formatting
instruction ("answer in all caps", "include exactly 3 bullet points") is a
shallow, mostly-lexical behaviour that survives 4-bit almost intact. If your
quantized model only ever gets an IFEval-style eval, you will not see the damage.

### 2. Perplexity ranked the two quantizers backwards

On perplexity, `self_awq` (+1.39) looks **better** than `mlx_awq_int4` (+1.60) —
lower is better, and my number is lower. On GSM8K the order **reverses and
widens**: `self_awq` loses 9.5 points, `mlx_awq_int4` loses 4.0. A 0.2-ppl
advantage on wikitext corresponds to a 5.5-point *deficit* on grade-school math.

Perplexity is an average per-token surprise over prose. It is dominated by the
easy, high-probability tokens. Two 4-bit models can land within 0.2 ppl of each
other and still differ by 5+ points on a task that depends on the model getting
a specific chain of low-probability tokens *all* right. The ppl gap and the
reasoning gap are not measuring the same thing, and here they don't even agree on
sign.

### Why the two AWQ arms differ

They are both "AWQ 4-bit, group size 128", but:

- `mlx_lm.awq` runs the **full AWQ search** — per-channel scale *and* a separate
  weight-clipping search (`--n-grid 10`). My `search_scale` does the scale search
  only.
- `mlx_lm.awq` quantizes every one of the 196 Linear layers plus the embeddings,
  with **no fp16 fallback**. Mine keeps **6 layers fp16** (the ones the search
  couldn't improve, including sensitive `down_proj` / `o_proj`) and leaves the
  embeddings fp16.
- Different calibration sets (mine: 32 rows of wikitext; mlx's: its bundled
  default).

None of that is the fake-vs-real-kernel distinction from [note 05](05-fake-quant-vs-real-int4.md)
— fake-quant, if anything, is *more* faithful because it skips the kernel's
rounding. The gap is entirely in the scale/clip search and the calibration, and
those details barely register on perplexity while moving GSM8K by 5 points.
"AWQ" is not one number.

## Pitfalls hit along the way

- **strict-match is unreliable with a chat model** (坑22). GSM8K's `strict-match`
  filter wants the answer at a fixed position (`#### 42`); Qwen's chat template
  produces a chatty CoT that ends "...so the total is **72**." fp16 itself scores
  only 0.378 strict vs 0.648 flexible. The strict-match *deltas* (−22, −17 pts)
  are mostly measuring format compliance, not arithmetic. `flexible-extract`
  (last number in the response) is the metric to report; strict-match is noise
  here. This is a cousin of lm-eval issue #1841 (chat templates silently move
  scores).
- **Confirm the requests actually arrive** (Bullet 3 pitfall 3). A misconfigured
  endpoint returns 200 OK with an empty completion and the score sits near
  random. The harness logs each arm's server: 801 `POST /v1/chat/completions`
  (400 + 400 + 1 warm-up) for all three arms — so the numbers are real.
- **loglikelihood tasks don't run over a chat API** (Bullet 3 pitfall 1). MMLU
  and HellaSwag score by comparing prompt logprobs of candidate continuations;
  `/v1/chat/completions` doesn't expose those. GSM8K and IFEval are generative,
  so they work. MMLU on this stack would need `local-completions` against a
  server that returns `logprobs` — out of scope here.
- **Give CoT room** (Bullet 3 pitfall 4). GSM8K's default `max_gen_toks` (256)
  truncates some chains; bumped to 768.

## The lesson

Perplexity is a screening metric, not an acceptance metric. It is cheap, it
correlates *loosely* with quality, and it will happily tell you that the quantizer
with the worse reasoning is the better one. Before shipping a quantized model,
run it on at least one task that requires a correct multi-step output — and hold
the eval config (chat template, few-shot format, decoding params) **identical**
between the baseline and the quantized model, because those knobs move the score
as much as the quantization does.
