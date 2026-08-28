# Fake-quant vs real int4: the number you report depends on the runtime you can't run

**Tasks:** P2.1–P2.4 (fake-quant AWQ/GPTQ study) → P6.2 (real int4 on Apple
silicon).
**Pitfall:** 坑21.

## Two different things both called "int4"

Most of the quantization study (支柱2) runs **fake quantization**: weights are
rounded to a 4-bit grid, then stored and multiplied as fp16. The arithmetic is
still fp16 matmul on the MPS backend. This is the right tool for studying
*quantization error* — calibration-set effects, cross-distribution robustness,
AWQ vs GPTQ — because it isolates the rounding from the kernel. It tells you
nothing about speed or memory, because nothing is actually 4 bits at runtime.

**Real int4** means 4-bit weights unpacked inside a fused Metal kernel, with the
memory and bandwidth win that implies. On this Mac there is exactly one runtime
that does it: `mlx-lm`. PyTorch's quantized backend was never ported to MPS
(project plan §9.1 Risk B), so the torch path *cannot* give a real-int4 number
no matter how the config is written. The runtime you have access to decides
which measurement you're even able to make.

## P6.2: the 4-way local comparison

Qwen2.5-1.5B-Instruct, bits=4, group-size=128, all on this machine, all sharing
one wikitext-2 token array so the deltas are internally consistent:

| arm | wikitext-2 ppl | Δ vs fp16 | decode tok/s | weights |
|-----|---------------:|----------:|-------------:|--------:|
| fp16 (mlx)         | 11.54 | — | 31 | 3.09 GB |
| **AWQ int4** (mlx) | 13.14 | **+1.60** | 104 | 0.84 GB |
| RTN int4 (mlx)     | 13.81 | +2.26 | 100 | 0.85 GB |
| GPTQ int4 (mlx)    | — | **DEGENERATE** | 99 | 0.91 GB |

Headline: real int4 shrinks the weights **3.7×** and speeds decode **3.3×**, for
**+1.6 ppl**. AWQ's calibration buys **+0.66 ppl** over naive round-to-nearest —
that's the concrete value of activation-aware scaling on this model.

Compare to the fake-quant AWQ result from 支柱2: **+1.2 ppl** (torch
sliding-window harness, fp16 baseline 12.18). The real-int4 penalty is
*somewhat larger* (+1.6 vs +1.2), but the harnesses differ — non-overlapping mlx
blocks vs torch sliding window, different fp16 baselines — so the two numbers are
**not directly comparable** and I don't claim a precise "kernel adds 0.4 ppl"
delta. The honest statement is: fake-quant is a lower bound on the error; the
real kernel's extra rounding in the fused path pushes it up modestly.

## The GPTQ degeneracy (坑21)

`mlx_lm.gptq` at bits=4 g=128 on this model produced a model that emits a
constant `!` and whose forward NLL is `nan`. First hypothesis was calibration
starvation — the first run gave GPTQ only 2 Hessian batches from
`--num-samples 16`. Re-ran at `--num-samples 64 --sequence-length 512`: **still
degenerate**. `mlx_lm.awq` and `mlx_lm.convert -q` RTN on the identical
model/config are both fine. I didn't bisect further (group size? a bad fallback
layer? an mlx-lm bug for the Qwen2 architecture?) — that's out of compute budget
and AWQ was already the primary arm.

Two engineering responses:

1. **Sentinel-check borrowed artifacts.** A third-party quantizer "succeeding"
   means the process exited 0 and a weights file exists. That is not the same as
   the model working. The bar is: generate one sentence — are they words? Run one
   forward — is the NLL finite? Only then trust it as a comparison arm.
2. **Keep the degenerate arm in the results.** "`mlx_lm.gptq` 0.31.3 does not
   produce a usable model for this architecture on Apple silicon" is a useful
   finding for anyone else trying this path. The result file's
   `quant_config.gptq` documents the full repro; the harness sanitizes the
   non-finite ppl to `null` + `degenerate_forward: true`, keeps the JSON strict
   (no `NaN` token), records `None` for the delta, and prints `GPTQ DEGENERATE`
   in the headline rather than crashing or silently dropping the arm.

## The lesson

- Know which measurement your runtime can actually make. "int4" in a config
  field is not int4 at runtime unless the kernel exists for your backend.
- Fake-quant perplexity is a lower bound on real-int4 perplexity, and cross-
  harness deltas are not subtraction-comparable — say so instead of implying more
  precision than the setup supports.
- A quantized model from someone else's tool gets a "can it form a sentence"
  check before it's allowed into your comparison. And when it fails that check,
  the failure is a row in the results table, not an exception.
