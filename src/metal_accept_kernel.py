"""
P6.7 -- a fused Metal kernel for the speculative-decoding accept/reject step.

The one primitive that is unique to speculative decoding (every other op -- the
matmuls, the norms, RoPE, the KV cache -- is shared with plain autoregressive
decoding) is the *rejection-sampling verification* step: given the target and
draft next-token distributions at the gamma proposed positions, decide how many
drafts to accept and, on the first rejection, build the adjusted distribution
    p'(x) = norm(max(0, p_target(x) - p_draft(x)))
to resample from (Leviathan et al. 2023; see src/rejection_sampling.py for the
scalar reference).

`src/rejection_sampling.py` does this as a Python scalar loop over gamma with
per-row `torch.softmax` -- fine for a few-token round, obviously correct, and the
path every correctness test exercises. This module asks a different question:
*if* this step were on the critical path (large vocab, batched verification),
what does it cost, and can one fused Metal kernel beat the naive MLX op graph?

Three MLX contenders + one Metal kernel, all fp32, same inputs, same outputs
`(n_accepted: int, adjusted_row: [V] float32)`:

  reference_sync        -- softmax x2, gather, accept scan (needs an eval to learn
                           n_accepted), then build the one adjusted row.
  reference_branchless  -- same, but compute all gamma+1 candidate adjusted rows
                           and select -- pure lazy graph, no mid-op sync.
  reference_compiled    -- reference_branchless wrapped in mx.compile.
  fused_accept          -- one mx.fast.metal_kernel: one threadgroup, one-pass
                           online-softmax stats per row, the accept scan on
                           thread 0, then the adjusted row. Reads the logit
                           tensors from device memory but writes no [*, V]
                           intermediate.

Roofline: V = 151936 (Qwen2.5), gamma ~ 4-8. The op is ~10 exp/FMA per logit
against one 4-byte read: arithmetic intensity < 1 flop/byte, i.e. deep in the
memory-bound regime on any Apple GPU (measured streaming BW ~84 GB/s on this
Mac). The only lever is bytes moved; see verify_p6_7_metal_roofline.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

VOCAB_QWEN25 = 151936
_TG = 1024  # threads per (single) threadgroup; must match the launch below


# --------------------------------------------------------------------------- #
# byte / flop accounting -- used by the roofline driver and the unit tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Traffic:
    bytes_moved: int
    flops: int

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes_moved


def fused_traffic(gamma: int, vocab: int) -> Traffic:
    """Bytes the fused kernel *must* move (device<->GPU), ignoring cache reuse.

    reads : target_logits (gamma+1, V) once for stats + once for the residual row,
            draft_logits  (gamma,   V) once for stats + once for the residual row.
            (the online-softmax pass is single-pass; the residual pass re-reads.)
    writes: adjusted (V).
    """
    g1 = gamma + 1
    reads = (2 * g1 * vocab + 2 * gamma * vocab) * 4
    writes = vocab * 4
    # ~ (exp + fma) per logit on the stats pass, (exp + exp + sub + max) on the
    # residual pass; count generously at 12 flop/logit over the touched logits.
    flops = 12 * (g1 + gamma) * vocab
    return Traffic(reads + writes, flops)


def reference_branchless_traffic(gamma: int, vocab: int) -> Traffic:
    """Bytes the naive lazy graph moves if *no* fusion happened: every elementwise
    [*, V] result is a full write + a full read by the next op.

    softmax(target) : read + write   (g1, V)          -> treat as 2 passes
    softmax(draft)  : read + write   (g,  V)
    all g+1 residual rows : read pt (g1,V) + read pd broadcast (g1,V)
                           + write (g1, V)
    normalise            : read + write (g1, V)
    select 1 row         : read (g1, V) + write (V)
    This is an upper bound; MLX fuses some of it (that is the point of the note).
    """
    g1 = gamma + 1
    words = (
        2 * g1 * vocab          # softmax target
        + 2 * gamma * vocab     # softmax draft
        + 3 * g1 * vocab        # residual rows: pt + pd + out
        + 2 * g1 * vocab        # normalise
        + g1 * vocab + vocab    # select
    )
    flops = 15 * g1 * vocab
    return Traffic(words * 4, flops)


# --------------------------------------------------------------------------- #
# MLX reference implementations
# --------------------------------------------------------------------------- #
def _accept_scan(pt_x: mx.array, pd_x: mx.array, unif: mx.array) -> int:
    """Leading run of accepted drafts. Tiny (gamma ~ 8); evaluated on the host."""
    a = mx.where(pd_x <= 0.0, mx.array(1.0), mx.minimum(1.0, pt_x / mx.maximum(pd_x, 1e-30)))
    accepted = mx.array(unif) < a
    acc = [bool(x) for x in accepted.tolist()]
    n = 0
    for ok in acc:
        if ok:
            n += 1
        else:
            break
    return n


def reference_sync(tl: mx.array, dl: mx.array, toks: mx.array, unif: mx.array):
    """softmax x2 -> gather -> scan (host sync) -> build the single adjusted row."""
    g1 = tl.shape[0]
    g = g1 - 1
    pt = mx.softmax(tl.astype(mx.float32), axis=-1)
    pd = mx.softmax(dl.astype(mx.float32), axis=-1)
    idx = mx.arange(g)
    pt_x = pt[idx, toks]
    pd_x = pd[idx, toks]
    mx.eval(pt_x, pd_x, pt, pd)
    n_acc = _accept_scan(pt_x, pd_x, unif)
    if n_acc == g:
        adjusted = pt[g]
    else:
        res = mx.maximum(0.0, pt[n_acc] - pd[n_acc])
        tot = mx.sum(res)
        adjusted = mx.where(tot < 1e-12, pt[n_acc], res / tot)
    mx.eval(adjusted)
    return n_acc, adjusted


def _all_adjusted_rows(pt: mx.array, pd: mx.array) -> mx.array:
    """[(g+1), V] -- candidate adjusted row for every possible n_accepted j.

    j < g : norm(max(0, pt[j] - pd[j]))   (fallback to pt[j] if residual ~ 0)
    j = g : pt[g]                          (the bonus row)
    """
    g1 = pt.shape[0]
    g = g1 - 1
    res = mx.maximum(0.0, pt[:g] - pd[:g])             # [g, V]
    tot = mx.sum(res, axis=-1, keepdims=True)          # [g, 1]
    norm = mx.where(tot < 1e-12, pt[:g], res / mx.maximum(tot, 1e-30))
    return mx.concatenate([norm, pt[g:g1]], axis=0)    # [g+1, V]


def reference_branchless(tl: mx.array, dl: mx.array, toks: mx.array, unif: mx.array):
    """No mid-op sync: build every candidate adjusted row, then select."""
    g1 = tl.shape[0]
    g = g1 - 1
    pt = mx.softmax(tl.astype(mx.float32), axis=-1)
    pd = mx.softmax(dl.astype(mx.float32), axis=-1)
    idx = mx.arange(g)
    a = mx.where(
        pd[idx, toks] <= 0.0,
        mx.array(1.0),
        mx.minimum(1.0, pt[idx, toks] / mx.maximum(pd[idx, toks], 1e-30)),
    )
    accepted = (mx.array(unif) < a).astype(mx.int32)          # [g]
    n_acc = mx.sum(mx.cumprod(accepted, axis=0))              # leading run of 1s
    rows = _all_adjusted_rows(pt, pd)                         # [g+1, V]
    adjusted = mx.take(rows, n_acc[None], axis=0)[0]          # [V]
    mx.eval(n_acc, adjusted)
    return int(n_acc.item()), adjusted


_compiled = None


def reference_compiled(tl: mx.array, dl: mx.array, toks: mx.array, unif: mx.array):
    global _compiled
    if _compiled is None:
        def _f(tl, dl, toks, unif):
            g1 = tl.shape[0]
            g = g1 - 1
            pt = mx.softmax(tl.astype(mx.float32), axis=-1)
            pd = mx.softmax(dl.astype(mx.float32), axis=-1)
            idx = mx.arange(g)
            a = mx.where(
                pd[idx, toks] <= 0.0,
                mx.array(1.0),
                mx.minimum(1.0, pt[idx, toks] / mx.maximum(pd[idx, toks], 1e-30)),
            )
            accepted = (mx.array(unif) < a).astype(mx.int32)
            n_acc = mx.sum(mx.cumprod(accepted, axis=0))
            rows = _all_adjusted_rows(pt, pd)
            adjusted = mx.take(rows, n_acc[None], axis=0)[0]
            return n_acc, adjusted
        _compiled = mx.compile(_f)
    n_acc, adjusted = _compiled(tl, dl, toks, unif)
    mx.eval(n_acc, adjusted)
    return int(n_acc.item()), adjusted


# --------------------------------------------------------------------------- #
# The fused Metal kernel
# --------------------------------------------------------------------------- #
_KERNEL_SRC = r"""
    // one threadgroup, TG threads, strided over V. gamma <= 15.
    uint tid = thread_position_in_threadgroup.x;
    uint nt  = threads_per_threadgroup.x;

    const uint G  = gamma;
    const uint G1 = G + 1u;
    const uint V  = vocab;

    threadgroup float red_m[TG];
    threadgroup float red_s[TG];
    threadgroup float mt[16];
    threadgroup float st[16];
    threadgroup float md[16];
    threadgroup float sd[16];
    threadgroup int   nacc_sh[1];

    // ---- phase A: one-pass online-softmax stats for every row -------------
    for (uint r = 0; r < G1 + G; ++r) {
        const device float* row =
            (r < G1) ? (target_logits + r * V)
                     : (draft_logits + (r - G1) * V);
        float m = -INFINITY;
        float s = 0.0f;
        for (uint v = tid; v < V; v += nt) {
            float x = row[v];
            float m_new = max(m, x);
            s = s * exp(m - m_new) + exp(x - m_new);
            m = m_new;
        }
        red_m[tid] = m;
        red_s[tid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float mA = red_m[tid],        sA = red_s[tid];
                float mB = red_m[tid + stride], sB = red_s[tid + stride];
                float mM = max(mA, mB);
                // both lanes idle (nt > V): mM == -INFINITY -> exp(-inf - -inf) = NaN.
                // guard it; the merged (m, s) stays the identity (-inf, 0).
                float sM = (mM > -INFINITY)
                    ? (sA * exp(mA - mM) + sB * exp(mB - mM))
                    : 0.0f;
                red_m[tid] = mM;
                red_s[tid] = sM;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            if (r < G1) { mt[r] = red_m[0]; st[r] = red_s[0]; }
            else        { md[r - G1] = red_m[0]; sd[r - G1] = red_s[0]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- phase B: accept scan on thread 0 --------------------------------
    if (tid == 0) {
        int nacc = 0;
        for (uint i = 0; i < G; ++i) {
            uint tok = (uint) draft_tokens[i];
            float pt = exp(target_logits[i * V + tok] - mt[i]) / st[i];
            float pd = exp(draft_logits[i * V + tok]  - md[i]) / sd[i];
            float a  = (pd <= 0.0f) ? 1.0f : min(1.0f, pt / pd);
            if (unif[i] < a) nacc += 1; else break;
        }
        nacc_sh[0] = nacc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int j = nacc_sh[0];
    if (tid == 0) n_accepted[0] = j;

    // ---- phase C: the one adjusted row ---------------------------------
    if (j == (int) G) {
        // bonus: target row G, already a normalised softmax
        for (uint v = tid; v < V; v += nt)
            adjusted[v] = exp(target_logits[G * V + v] - mt[G]) / st[G];
        return;
    }

    float lsum = 0.0f;
    for (uint v = tid; v < V; v += nt) {
        float pt = exp(target_logits[j * V + v] - mt[j]) / st[j];
        float pd = exp(draft_logits[j * V + v]  - md[j]) / sd[j];
        float rres = max(0.0f, pt - pd);
        adjusted[v] = rres;            // stash unnormalised
        lsum += rres;
    }
    red_s[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_s[tid] += red_s[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = red_s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (total < 1e-12f) {
        for (uint v = tid; v < V; v += nt)
            adjusted[v] = exp(target_logits[j * V + v] - mt[j]) / st[j];
    } else {
        for (uint v = tid; v < V; v += nt)
            adjusted[v] = adjusted[v] / total;
    }
"""

_kernel = mx.fast.metal_kernel(
    name="spec_accept_reject",
    input_names=["target_logits", "draft_logits", "draft_tokens", "unif", "gamma", "vocab"],
    output_names=["n_accepted", "adjusted"],
    header=f"#define TG {_TG}\n",
    source=_KERNEL_SRC,
)


def fused_accept(tl: mx.array, dl: mx.array, toks: mx.array, unif: mx.array):
    g1, v = tl.shape
    g = g1 - 1
    n_acc, adjusted = _kernel(
        inputs=[
            tl.astype(mx.float32),
            dl.astype(mx.float32),
            toks.astype(mx.int32),
            unif.astype(mx.float32),
            mx.array(g, dtype=mx.uint32),
            mx.array(v, dtype=mx.uint32),
        ],
        grid=(_TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(1,), (v,)],
        output_dtypes=[mx.int32, mx.float32],
    )
    mx.eval(n_acc, adjusted)
    return int(n_acc.item()), adjusted


IMPLS = {
    "reference_sync": reference_sync,
    "reference_branchless": reference_branchless,
    "reference_compiled": reference_compiled,
    "fused_accept": fused_accept,
}


# --------------------------------------------------------------------------- #
# synthetic inputs
# --------------------------------------------------------------------------- #
def make_inputs(gamma: int, vocab: int, seed: int = 0, accept_frac: float = 0.7):
    """target/draft logits with a controllable expected acceptance rate.

    draft_logits ~ N(0,1); target_logits = draft_logits + N(0, spread) so the two
    distributions overlap by roughly `accept_frac`. draft_tokens are drawn from
    the draft distribution (as the algorithm does). `unif` fixed per position.
    """
    rng = mx.random.key(seed)
    k1, k2, k3, k4 = mx.random.split(rng, 4)
    g1 = gamma + 1
    dl = mx.random.normal((g1, vocab), key=k1)
    spread = 0.4 + (1.0 - accept_frac)  # more spread -> lower acceptance
    tl = dl + spread * mx.random.normal((g1, vocab), key=k2)
    pd = mx.softmax(dl[:gamma], axis=-1)
    toks = mx.array(
        [int(mx.random.categorical(mx.log(pd[i][None]), key=mx.random.split(k3, gamma)[i]).item())
         for i in range(gamma)],
        dtype=mx.int32,
    )
    unif = mx.random.uniform(shape=(gamma,), key=k4)
    mx.eval(tl, dl, toks, unif)
    return tl, dl, toks, unif
