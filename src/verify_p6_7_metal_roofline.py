"""
P6.7 -- roofline case study: one fused Metal kernel vs the naive MLX op graph
for the speculative-decoding accept/reject step.

project_plan_v9.md sec 7 P6.7 / 支柱7 (optional). This is the "write a custom
Metal kernel and put it on a roofline" item. The honest prior is that MLX's own
lazy-graph fusion is already good and mx.fast already wins the ops that matter;
this driver's job is to *measure* that rather than assert it, on the one op that
is actually unique to speculative decoding.

What it does, all on-GPU, fp32, V = 151936 (Qwen2.5):

  1. correctness    -- fused_accept vs reference_branchless over a (gamma x
                       accept_frac x seed) grid: n_accepted exact, adjusted row
                       within tol.
  2. device peaks   -- streaming bandwidth (big a+1) and fp32 GFLOP/s (big
                       square matmul), measured in-process.
  3. latency        -- median us/call for each impl at gamma in {2,4,8},
                       warmup then timed, mx.synchronize between.
  4. roofline       -- arithmetic intensity (flop/byte) and achieved bandwidth
                       for each impl; the ridge point; which side each sits on.

Writes results/p6_7_metal_roofline.json.

Run:
    .venv/bin/python src/verify_p6_7_metal_roofline.py           # ~1-2 min
    .venv/bin/python src/verify_p6_7_metal_roofline.py --smoke   # tiny, seconds
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx

from metal_accept_kernel import (
    IMPLS,
    VOCAB_QWEN25,
    fused_traffic,
    make_inputs,
    reference_branchless_traffic,
)

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "results" / "p6_7_metal_roofline.json"


# --------------------------------------------------------------------------- #
def measure_peak_bandwidth(nbytes: int = 1 << 28, iters: int = 30) -> float:
    """GB/s for a streaming a + 1 (read n, write n)."""
    n = nbytes // 4
    a = mx.random.normal((n,))
    mx.eval(a)
    b = a + 1.0
    mx.eval(b)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b = a + 1.0
        mx.eval(b)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (a.nbytes + b.nbytes) / dt / 1e9


def measure_peak_gflops(n: int = 4096, iters: int = 30) -> float:
    """fp32 GFLOP/s for a big square matmul (2 n^3 flops)."""
    a = mx.random.normal((n, n))
    bb = mx.random.normal((n, n))
    mx.eval(a, bb)
    c = a @ bb
    mx.eval(c)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ bb
        mx.eval(c)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (2.0 * n ** 3) / dt / 1e9


# --------------------------------------------------------------------------- #
def check_correctness(vocab: int, tol: float, smoke: bool) -> dict:
    gammas = (4,) if smoke else (2, 4, 6, 8)
    fracs = (0.6,) if smoke else (0.9, 0.6, 0.3)
    seeds = (0,) if smoke else (0, 1, 2)
    n_exact = 0
    n_total = 0
    max_adj_diff = 0.0
    worst = None
    for g in gammas:
        for af in fracs:
            for sd in seeds:
                tl, dl, toks, unif = make_inputs(g, vocab, seed=sd * 97 + g * 7 + int(af * 10), accept_frac=af)
                ref_n, ref_adj = IMPLS["reference_branchless"](tl, dl, toks, unif)
                fus_n, fus_adj = IMPLS["fused_accept"](tl, dl, toks, unif)
                n_total += 1
                n_exact += int(ref_n == fus_n)
                d = float(mx.max(mx.abs(fus_adj - ref_adj)))
                if d > max_adj_diff:
                    max_adj_diff = d
                    worst = {"gamma": g, "accept_frac": af, "seed": sd, "ref_n": ref_n, "fused_n": fus_n}
    return {
        "cases": n_total,
        "n_accepted_exact": f"{n_exact}/{n_total}",
        "max_adjusted_abs_diff": max_adj_diff,
        "tolerance": tol,
        "within_tol": bool(max_adj_diff <= tol),
        "worst_case": worst,
    }


def bench_impl(fn, tl, dl, toks, unif, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(tl, dl, toks, unif)
    mx.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(tl, dl, toks, unif)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def run(smoke: bool = False) -> dict:
    vocab = 4096 if smoke else VOCAB_QWEN25
    iters = 20 if smoke else 200
    warmup = 3 if smoke else 20
    gammas = (4,) if smoke else (2, 4, 8)

    peak_bw = measure_peak_bandwidth(1 << 22 if smoke else 1 << 28, 5 if smoke else 30)
    peak_gflops = measure_peak_gflops(512 if smoke else 4096, 5 if smoke else 30)
    ridge_point = peak_gflops / peak_bw  # flop/byte where compute-bound begins

    correctness = check_correctness(vocab, tol=1e-6, smoke=smoke)

    per_gamma = {}
    for g in gammas:
        tl, dl, toks, unif = make_inputs(g, vocab, seed=g, accept_frac=0.6)
        fused_t = fused_traffic(g, vocab)
        naive_t = reference_branchless_traffic(g, vocab)
        rows = {}
        for name, fn in IMPLS.items():
            med = bench_impl(fn, tl, dl, toks, unif, iters, warmup)
            traffic = fused_t if name == "fused_accept" else naive_t
            achieved_bw = traffic.bytes_moved / med / 1e9
            rows[name] = {
                "median_us": med * 1e6,
                "calls_per_s": 1.0 / med,
                "model_bytes_moved": traffic.bytes_moved,
                "model_flops": traffic.flops,
                "arith_intensity_flop_per_byte": traffic.intensity,
                "achieved_GBps": achieved_bw,
                "achieved_vs_peak_bw": achieved_bw / peak_bw,
                "memory_bound": traffic.intensity < ridge_point,
            }
        fastest = min(rows, key=lambda k: rows[k]["median_us"])
        rows["_fused_speedup_vs_best_reference"] = (
            min(rows[k]["median_us"] for k in rows if k != "fused_accept" and not k.startswith("_"))
            / rows["fused_accept"]["median_us"]
        )
        rows["_fastest"] = fastest
        per_gamma[f"gamma={g}"] = rows

    # context: how big is this op next to the forward it hangs off of?
    # fp16 Qwen2.5-1.5B decode = 31.37 tok/s (results/p6_2_awq_int4_real.json),
    # i.e. ~31.9 ms per target forward at batch 1. A speculative verification
    # forward (g+1 positions in one pass) is the same order.
    fwd_ms = 1000.0 / 31.365505
    g4 = per_gamma.get("gamma=4") or next(iter(per_gamma.values()))
    accept_ms = min(r["median_us"] for k, r in g4.items() if not k.startswith("_")) / 1e3
    context = {
        "target_forward_ms_batch1": fwd_ms,
        "target_forward_source": "results/p6_2_awq_int4_real.json fp16_mlx decode_tok_s.mean=31.37",
        "best_accept_step_ms_gamma4": accept_ms,
        "accept_step_fraction_of_one_forward": accept_ms / fwd_ms,
        "note": "temperature>0 path; greedy decoding replaces the two softmaxes with "
                "an argmax and the accept step is effectively free.",
    }

    return {
        "task": "P6.7 fused Metal accept/reject kernel -- roofline case study",
        "op": "speculative-decoding rejection-sampling verification: (target_logits[g+1,V], "
              "draft_logits[g,V], draft_tokens[g], unif[g]) -> (n_accepted, adjusted_row[V])",
        "config": {
            "vocab": vocab,
            "dtype": "float32",
            "iters": iters,
            "warmup": warmup,
            "threadgroup": 1024,
            "single_threadgroup": True,
        },
        "device": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "mlx_version": mx.__version__,
            "measured_peak_streaming_GBps": peak_bw,
            "measured_peak_fp32_GFLOPs": peak_gflops,
            "roofline_ridge_point_flop_per_byte": ridge_point,
        },
        "correctness": correctness,
        "context_vs_model_forward": context,
        "latency_and_roofline": per_gamma,
        "headline": _headline(per_gamma, context, ridge_point),
    }


def _headline(per_gamma: dict, context: dict, ridge_point: float) -> str:
    g4 = per_gamma.get("gamma=4") or next(iter(per_gamma.values()))
    sp = g4["_fused_speedup_vs_best_reference"]
    fused = g4["fused_accept"]
    ref = g4["reference_compiled"]
    byte_ratio = ref["model_bytes_moved"] / fused["model_bytes_moved"]
    fused_occ = fused["achieved_vs_peak_bw"]
    ref_occ = ref["achieved_vs_peak_bw"]
    frac = context["accept_step_fraction_of_one_forward"]
    rel = "faster than" if sp > 1.02 else ("on par with" if sp > 0.95 else "slower than")
    return (
        f"At gamma=4 the fused kernel moves {byte_ratio:.1f}x fewer bytes than the naive "
        f"model yet runs {sp:.2f}x ({rel}) the mx.compile'd reference. The roofline says "
        f"memory-bound (arith intensity ~ 0.4-1.5 flop/byte vs ridge ~ {ridge_point:.0f}), "
        f"but a single-threadgroup kernel tops out at ~{fused_occ*100:.0f}% of peak "
        f"bandwidth while MLX fans the op across every GPU core and hits ~{ref_occ*100:.0f}%: "
        f"less traffic x worse occupancy = a wash. And the whole step is only "
        f"~{frac*100:.1f}% of one target forward, and zero under greedy. mx.compile on the "
        f"plain op graph is the right answer."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=JSON_PATH)
    args = ap.parse_args()

    t0 = time.perf_counter()
    result = run(smoke=args.smoke)
    result["total_seconds"] = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
