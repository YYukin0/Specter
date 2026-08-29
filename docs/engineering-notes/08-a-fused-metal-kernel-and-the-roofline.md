# A fused Metal accept/reject kernel: 2.6× less memory traffic, no speedup

**Task:** P6.7 — hand-write one Metal kernel for the speculative-decoding
accept/reject step, put it on a roofline, and find out whether beating MLX's own
op graph is possible or worth it.
**Pitfall:** 坑24.
**Code:** `src/metal_accept_kernel.py`, `src/verify_p6_7_metal_roofline.py`,
`results/p6_7_metal_roofline.json`.

## Why this op

Almost nothing in a speculative decoder is *its own* code. The matmuls, RMSNorm,
RoPE, the KV cache, softmax, sampling — all of it is shared with plain
autoregressive decoding and all of it is already serviced by `mx.fast` /
`mlx-lm`. The one primitive that only exists because of speculative decoding is
the **rejection-sampling verification step**: given the target and draft
next-token distributions at the γ proposed positions, decide how many drafts to
accept, and on the first rejection build the adjusted distribution

```
p'(x) = norm(max(0, p_target(x) − p_draft(x)))
```

to resample from (Leviathan et al. 2023). `src/rejection_sampling.py` does this
as a Python scalar loop over γ with a `torch.softmax` per row — obviously
correct, and the path every correctness test exercises. If any hand-written
kernel in this repo is going to earn its place, it's this one. So: what does the
step cost as a batched GPU op, and can a single fused kernel beat the naive MLX
version?

## What I built

`fused_accept` — one `mx.fast.metal_kernel`, one threadgroup of 1024 threads:

- **one-pass online softmax** for every row (running max + running sum-of-exp in
  a single sweep over V, then a threadgroup merge), so the softmax stats cost one
  read of the logits, not two;
- the **accept scan** on thread 0 (γ ≤ 15, it's nothing);
- **one adjusted row** written out — the residual at the first rejected position,
  or the target row at the bonus slot — normalised in place.

Crucially it writes **no `[*, V]` intermediate**. The naive MLX path
(`reference_branchless`: two `mx.softmax`, gather, build all γ+1 candidate rows,
select) materialises several. By my byte model that's **2.6× less memory traffic
at γ=4** (11.5 MB vs 29.8 MB).

Three MLX contenders for a fair fight: `reference_sync` (host sync to learn
`n_accepted`, then build the one row), `reference_branchless` (pure lazy graph),
and `reference_compiled` (`mx.compile` of the branchless version).

## The result

Qwen2.5 vocab V = 151936, fp32, this 24 GB Mac (measured streaming BW ~84 GB/s,
fp32 GFLOP/s ~3050, roofline ridge point ≈ 33–36 flop/byte). Median µs/call from
one representative run (the op is small enough that Python/dispatch overhead puts
±3–5% of noise on every number):

| γ | reference_sync | reference_branchless | reference_compiled | **fused_accept** | fused ÷ best-ref |
|--:|---------------:|---------------------:|-------------------:|-----------------:|----------------:|
| 2 | 696 | 517 | 469 | **405** | ~1.1× |
| 4 | 900 | 829 | 741 | **729** | ~1.0× |
| 8 | 1030 | 1253 | 1100 | **1058** | ~1.0× |

Across runs the fused kernel lands between 0.97× and 1.16× of the best MLX
reference — i.e. a wash, edging ahead at small γ and falling behind by γ=8.
Correctness is stable: `n_accepted` exact in 36/36 (γ × accept-rate × seed)
cases; adjusted row within 1e-9 of the reference (fp32 reduction-order noise).

So: **2.6× fewer bytes, ~1.0× the speed.** `mx.compile` on the plain op graph is
the best MLX path and the fused kernel only ties it.

## Why less memory didn't buy less time

The roofline says this op is deep in the memory-bound regime — arithmetic
intensity 0.4–1.5 flop/byte against a ridge point of 36 — so bytes moved *should*
be the only thing that matters. It isn't, because the textbook roofline assumes
you **saturate** the bandwidth you're bound by.

Look at the achieved bandwidth:

| impl | achieved BW (γ=4) | % of 84 GB/s peak |
|------|------------------:|------------------:|
| reference_compiled | 40 GB/s | 48% |
| **fused_accept** | 16 GB/s | **19%** |

The fused kernel is **one threadgroup** — it runs on a single GPU core. MLX's
path is several library kernels, each of which fans across every core on the
chip. One core moving 2.6× less data at 2.5× worse bandwidth utilisation comes
out even. **Occupancy is the axis the roofline picture hides:** "memory-bound"
is a statement about the ceiling, not about whether a given kernel reaches it.

A fused kernel that actually won would need to split V across many threadgroups
with a two-level reduction (partial stats per group → global merge → second pass
for the residual row) — at which point I've re-implemented what MLX's kernel
generator already emits, and the `mx.compile` number says it emits it well.

## The op's share of total decode cost

Even a hypothetical 2× win here is **~2.3% of one target forward**. The fp16
1.5B target decodes at 31 tok/s (P6.2) — ~32 ms per forward — and the accept step
is 0.73 ms. Under **greedy** decoding it's less than that: no softmax at all,
just an argmax, and the whole "distribution" machinery collapses. This op is not
on the critical path and was never going to be.

That's the honest result the plan predicted before I started: on this stack, for
pointwise / small-reduction work, **`mx.compile` on a clean op graph is the right
answer**, and a hand-written Metal kernel is a learning exercise, not an
optimisation. Worth doing once to know where the line is.

## Pitfall hit along the way

- **NaN when the vocab is smaller than the threadgroup** (坑24). The one-pass
  online-softmax merge computes `mM = max(mA, mB)` then `sA·exp(mA−mM) +
  sB·exp(mB−mM)`. When V < 1024, the surplus threads never enter the reduction
  loop and carry the identity `(m, s) = (−∞, 0)`; two of them merging gives
  `mM = −∞` and `exp(−∞ − (−∞)) = exp(NaN) = NaN`, which then poisons the row.
  Invisible at the real V = 151936 (every thread does work), caught immediately
  by the V = 256 unit test — the reason the smoke config and the kernel tests
  deliberately use a tiny vocab. Fix: guard the merge with
  `mM > −INFINITY ? … : 0`.

## The lesson

The roofline tells you which resource caps a kernel; it does not tell you
whether *your* kernel reaches the cap. Before hand-writing a kernel, check two
things the roofline leaves out: can this launch use the whole GPU, and is the op
even a measurable fraction of the thing it hangs off? Here the answers were "no"
and "no", and `mx.compile` had already won.
