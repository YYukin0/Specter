"""
P6.1 verify -- SpecServer throughput vs concurrency, breaker ON/OFF, and the
EQSPEC realignment tax (computed for the rectangular padded-batch counterfactual,
not paid -- see src/spec_kv_batch.py).

Real Qwen2.5 0.5B/1.5B on MPS fp16. A fixed workload of 8 prompts is served at
max_active in {1,2,4,8}; the per-sequence caches mean output is identical at
every width (pinned bit-exactly on the FakeModel in tests/). What changes with
width is wall-clock and the ragged realignment overhead.

Reported per (context_regime, max_active, breaker):
  * agg_tok_per_s        total emitted tokens / wall-clock
  * per_seq_tok_per_s    mean over requests of (tokens / that request's span)
  * speedup_vs_max_active_1
  * realignment_overhead mean / p90 / max over rounds -- "if you batched the
    ragged verify into one rectangular tensor, this fraction would be padding"
    (deployment-depth-plan sec 7 C3: EQSPEC pays ~40% of this at BS=8)
  * rolling_alpha_mean, n_spec / n_degraded / n_probe rounds, mode_switches
Plus a KV-cached target-only baseline (the fair yardstick from P6.0).

Two context regimes (deployment-depth-plan sec 7 C4 -- big-batch speculation is
regime dependent): "short" = the 8 P6.x prompts as-is; "long" = each prefixed
with a ~400-token filler so the KV cache dominates.

Run:  python src/verify_serving_loop.py           (full, a few min)
      python src/verify_serving_loop.py --smoke    (max_active {1,4}, short only)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from prompts import PROMPTS
from serving_loop import ServeConfig, SpecServer
from spec_kv import target_only_generate_kv

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p6_1_serving_throughput.json"

GAMMA = 3
MAX_NEW_TOKENS = 48
WIDTHS = [1, 2, 4, 8]
_LONG_FILLER = (
    "The following is background context that should be read carefully before "
    "answering. " * 40
)


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


def _prompts(regime):
    if regime == "long":
        return [f"{_LONG_FILLER}\n\nQuestion: {p}" for p in PROMPTS]
    return list(PROMPTS)


def _serve_once(draft, target, tok, prompts, *, width, breaker_on):
    cfg = ServeConfig(
        gamma=GAMMA, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS, max_active=width,
        breaker_on=breaker_on, alpha_floor=0.5, warmup_rounds=4, reprobe_every=20,
    )
    srv = SpecServer(draft, target, tok, cfg)
    for i, p in enumerate(prompts):
        srv.submit(p, req_id=f"p{i}", seed=1000 + i)

    t0 = time.perf_counter()
    srv.run_until_idle(max_rounds=5000)
    wall = time.perf_counter() - t0

    res = srv.results()
    total_tokens = sum(len(r.token_ids) for r in res.values())
    round_walls = [inf.wall_s for inf in srv.round_log if inf.mode != "idle"]
    reali = [t.realignment_overhead for t in srv.telemetry]
    modes = [inf.mode for inf in srv.round_log]

    mean_round_wall = statistics.fmean(round_walls) if round_walls else 0.0
    per_seq_tok_per_s = []
    for r in res.values():
        span = max(1, r.finish_round - r.admit_round) * mean_round_wall
        per_seq_tok_per_s.append(len(r.token_ids) / span if span else 0.0)

    return {
        "width": width,
        "breaker": "on" if breaker_on else "off",
        "n_requests": len(res),
        "wall_s": wall,
        "rounds": len([m for m in modes if m != "idle"]),
        "agg_tok_per_s": total_tokens / wall if wall else 0.0,
        "per_seq_tok_per_s_mean": _mean_std(per_seq_tok_per_s)[0],
        "total_tokens": total_tokens,
        "realignment_overhead_mean": _mean_std(reali)[0],
        "realignment_overhead_p90": _pct(reali, 0.9),
        "realignment_overhead_max": max(reali) if reali else 0.0,
        "rolling_alpha_mean": _mean_std([inf.rolling_alpha for inf in srv.round_log
                                         if inf.mode == "spec"])[0],
        "n_spec_rounds": sum(1 for m in modes if m == "spec"),
        "n_degraded_rounds": sum(1 for m in modes if m == "degraded"),
        "n_probe_rounds": sum(1 for m in modes if m == "probe"),
        "mean_accept_len": _mean_std(
            [statistics.fmean(r.accept_lengths) for r in res.values() if r.accept_lengths]
        )[0],
    }


def _target_only_baseline(target, tok, prompts):
    t0 = time.perf_counter()
    total = 0
    for i, p in enumerate(prompts):
        r = target_only_generate_kv(p, target, tok, max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=1.0, seed=1000 + i)
        total += len(r.token_ids)
    wall = time.perf_counter() - t0
    return {"agg_tok_per_s": total / wall if wall else 0.0, "wall_s": wall, "total_tokens": total}


def run(widths, regimes, smoke=False):
    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)

    out = {
        "task": "P6.1 SpecServer -- continuous batching + real-signal circuit breaker",
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "device": "mps",
        "dtype": "float16",
        "gamma": GAMMA,
        "max_new_tokens": MAX_NEW_TOKENS,
        "workload_prompts": len(PROMPTS),
        "widths": widths,
        "regimes": {},
        "caveats": [
            "Per-sequence KV caches -> output is bit-identical at every width "
            "(pinned on the FakeModel in tests/test_spec_kv_batch.py). Only "
            "wall-clock and realignment_overhead move with width.",
            "realignment_overhead is the COUNTERFACTUAL cost of folding the "
            "ragged verify into one rectangular padded tensor "
            "(1 - sum(work)/(n*max(work))). EQSPEC pays ~40% of this at BS=8 "
            "(arXiv:2510.22876). This path avoids it by not batching the kernel "
            "-- which is why agg_tok_per_s is near-flat in width.",
            "Wall-clock on MPS fp16, single 24GB Mac. Circuit breaker trips on "
            "rolling alpha < 0.5 (+ optional latency probe), never on batch size "
            "alone (contrast P5.3 坑15).",
        ],
    }

    for regime in regimes:
        prompts = _prompts(regime)
        base = _target_only_baseline(target, tok, prompts)
        rows = []
        for w in widths:
            for breaker_on in ([False] if smoke else [False, True]):
                row = _serve_once(draft, target, tok, prompts, width=w, breaker_on=breaker_on)
                rows.append(row)
                print(f"[{regime}] width={w} breaker={'on' if breaker_on else 'off':3s} "
                      f"agg={row['agg_tok_per_s']:.1f} tok/s  "
                      f"realign_mean={row['realignment_overhead_mean']:.2f} "
                      f"alpha={row['rolling_alpha_mean']:.2f} "
                      f"deg={row['n_degraded_rounds']}")

        w1 = next((r for r in rows if r["width"] == widths[0] and r["breaker"] == "off"), None)
        for r in rows:
            r["speedup_vs_width1_off"] = (
                r["agg_tok_per_s"] / w1["agg_tok_per_s"] if w1 and w1["agg_tok_per_s"] else 0.0
            )
            r["speedup_vs_target_only_kv"] = (
                r["agg_tok_per_s"] / base["agg_tok_per_s"] if base["agg_tok_per_s"] else 0.0
            )
        out["regimes"][regime] = {"target_only_kv_baseline": base, "runs": rows}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    widths = [1, 4] if args.smoke else WIDTHS
    regimes = ["short"] if args.smoke else ["short", "long"]

    t0 = time.perf_counter()
    out = run(widths, regimes, smoke=args.smoke)
    out["elapsed_s_total"] = time.perf_counter() - t0

    if not args.smoke:
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {RESULTS_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
