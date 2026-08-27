"""
P2.3 -- calibration-set SIZE ablation for self-built 4-bit AWQ.

Reference: notes/project_plan_v9.md sec 7 P2.3; AWQ paper (Lin et al. 2023) sec 4.3
("AWQ needs a much smaller calibration set" -- its perplexity is roughly flat from
a handful of sequences upward, whereas GPTQ keeps improving / overfits).

We vary n_calib in {4, 8, 16, 32, 64}, calibration distribution fixed = natural
language (wikitext2), 3 shuffles per size (seeds 0/1/2), quantize the whole
model, measure wikitext2 perplexity. Look for:
  - an overfitting knee: does small-n_calib ppl sit noticeably higher?
  - variance blow-up: is the std across shuffles much larger at small n_calib?

*** 坑16 (why this file caps rows at CALIB_ROW_TOKENS instead of 512) ***
The first P2.3 run passed max_tokens_per_layer=512 into capture_all_layer_inputs
(the P2.2 default). That function stops capturing once EVERY target layer has
512 tokens -- which the first one or two 512-token wikitext rows already satisfy.
So n_calib in {8,16,32,64,128} all fed the model the *identical* first-512-token
pool and produced a bit-identical ppl (seed 0: 13.6015 for every size). The
"flat curve" was a capture artifact, not evidence for AWQ's small-calib claim.
Fix here: truncate each calib row to CALIB_ROW_TOKENS=64 tokens and set the
per-layer cap to n_calib * CALIB_ROW_TOKENS, so the pool genuinely scales with
n_calib (64 -> 4096 tokens/layer at the top of the grid). n_calib=128 is dropped
because 8192 tokens x ~509k summed in-features x 4 B ~= 17 GB of CPU-side capture
would thrash this 24 GB machine; going higher needs capture_all_layer_inputs
rewritten to quantize layer-by-layer and free as it goes (decision point for the
user, noted in the results file).

Michael's 3B reference (results/p2_awq_calibration_michael_3b.json,
p2_3_calibration_size_ablation): ppl 11.22 -> 11.27 across n_calib 4..N, ~flat.
Direction expected to match here; a spike at small n would contradict it.

Run:  python src/verify_p2_3_calib_size.py [--smoke]
Writes results/p2_3_calib_size.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from awq_perplexity import load_eval_corpus  # noqa: E402
from verify_p2_2_cross_distribution import (  # noqa: E402
    EVAL_WINDOW,
    MODEL,
    _mean_std,
    _quantize_fresh,
    _fp16_baseline,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "p2_3_calib_size.json"

N_CALIB_GRID = [4, 8, 16, 32, 64]
SHUFFLE_SEEDS = [0, 1, 2]
DEFAULT_MAX_WINDOWS = 60
CALIB_ROW_TOKENS = 64        # each calib sequence truncated to this; n_calib is then the sole knob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="n_calib grid {4,8}, 1 shuffle, 3 windows, first 12 layers")
    ap.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    args = ap.parse_args()

    grid = N_CALIB_GRID
    seeds = SHUFFLE_SEEDS
    max_windows = args.max_windows
    layers_limit = None
    if args.smoke:
        grid, seeds, max_windows, layers_limit = [4, 8], [0], 3, 12

    print("loading corpora ...", flush=True)
    nl_pool = load_eval_corpus("wikitext2")
    eval_corpora = {"wikitext2": nl_pool}

    t0 = time.time()
    print("fp16 baseline ...", flush=True)
    baseline = _fp16_baseline(eval_corpora, max_windows)
    print(f"  fp16 wikitext2 ppl {baseline['wikitext2']:.4f}", flush=True)

    by_size = {}
    fell_back_by_size = {}
    captured_tokens_by_size = {}
    for n_calib in grid:
        ppls = []
        fbs = []
        caps = []
        for seed in seeds:
            rng = random.Random(seed)
            shuffled = nl_pool[:]
            rng.shuffle(shuffled)
            calib_texts = shuffled[:n_calib]
            cap = n_calib * CALIB_ROW_TOKENS
            print(f"quantize n_calib={n_calib} seed={seed} "
                  f"(cap {cap} tok/layer, {CALIB_ROW_TOKENS} tok/row) ...", flush=True)
            ppl, summ = _quantize_fresh(calib_texts, max_windows, eval_corpora,
                                        layers_limit=layers_limit,
                                        max_tokens_per_layer=cap,
                                        max_seq_len=CALIB_ROW_TOKENS)
            got = summ.get("captured_tokens_per_layer_min", 0)
            print(f"  -> wikitext2 ppl {ppl['wikitext2']:.4f}  "
                  f"(fell_back {summ['n_fell_back']}, captured {got} tok/layer)", flush=True)
            ppls.append(ppl["wikitext2"])
            fbs.append(summ["n_fell_back"])
            caps.append(got)
        by_size[str(n_calib)] = _mean_std(ppls)
        by_size[str(n_calib)]["delta_vs_fp16"] = by_size[str(n_calib)]["mean"] - baseline["wikitext2"]
        fell_back_by_size[str(n_calib)] = fbs
        captured_tokens_by_size[str(n_calib)] = caps

    means = sorted((int(k), v["mean"]) for k, v in by_size.items())
    stds = sorted((int(k), v["std"]) for k, v in by_size.items())
    small_key, large_key = str(grid[0]), str(grid[-1])
    knee = by_size[small_key]["mean"] - by_size[large_key]["mean"]
    var_ratio = (by_size[small_key]["std"] / by_size[large_key]["std"]
                 if by_size[large_key]["std"] > 1e-9 else None)

    # sanity: did the knob actually move this time?
    all_caps = [c for cs in captured_tokens_by_size.values() for c in cs]
    knob_moved = max(all_caps) > min(all_caps) * 1.5 if all_caps else False

    verdict = (
        f"n_calib {small_key} vs {large_key}: ppl {by_size[small_key]['mean']:.4f} vs "
        f"{by_size[large_key]['mean']:.4f} (small - large = {knee:+.4f}). "
        + ("flat -> AWQ 'small calibration set is enough' reproduced"
           if abs(knee) <= 0.15 else
           "small-n_calib ppl noticeably higher -> possible overfit/underfill, inspect")
        + f". std ratio small/large = "
        + (f"{var_ratio:.2f}" if var_ratio is not None else "n/a")
        + ("" if (var_ratio is None or var_ratio <= 3.0)
           else " (variance blows up at small n_calib -- signal)")
        + (". CAUTION: captured tok/layer did NOT scale with n_calib -- capture cap "
           "still binding, treat curve as inconclusive" if not knob_moved else "")
    )

    out = {
        "task": "P2.3 calibration-set size ablation for self-built 4-bit AWQ",
        "model": MODEL,
        "reference": "notes/project_plan_v9.md sec7 P2.3; AWQ paper sec 4.3",
        "config": {"n_calib_grid": grid, "shuffle_seeds": seeds,
                   "calib_distribution": "wikitext2 (natural language, fixed)",
                   "calib_row_tokens": CALIB_ROW_TOKENS,
                   "per_layer_cap": "n_calib * calib_row_tokens (scales with n_calib)",
                   "eval_window": EVAL_WINDOW, "eval_stride": EVAL_WINDOW,
                   "eval_max_windows": max_windows, "layers_limit": layers_limit},
        "capture_cap_bug_note": (
            "first run used a fixed 512 tok/layer cap that saturated after 1-2 rows, "
            "making ppl bit-identical for every n_calib >= 8 (坑16). This run scales "
            "the cap with n_calib and truncates rows to 64 tokens; n_calib=128 dropped "
            "for memory (needs layer-sequential capture rewrite)."),
        "fp16_baseline_wikitext2_ppl": baseline["wikitext2"],
        "ppl_by_calib_size": by_size,
        "n_fell_back_by_size": fell_back_by_size,
        "captured_tokens_per_layer_by_size": captured_tokens_by_size,
        "capture_knob_actually_moved": knob_moved,
        "curve_mean": means,
        "curve_std": stds,
        "knee_small_minus_large": knee,
        "variance_ratio_small_over_large": var_ratio,
        "verdict": verdict,
        "michael_3b_reference_direction_only": {
            "source": "results/p2_awq_calibration_michael_3b.json p2_3_calibration_size_ablation",
            "his_numbers": "ppl ~11.22 -> 11.27 across n_calib 4..N (flat) @ Qwen2.5-3B bf16",
            "his_setup_differs": "3B not 1.5B, bf16 not fp16, his own AWQ impl, longer calib rows",
            "our_direction_matches": bool(abs(knee) <= 0.2 and knob_moved),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(verdict, flush=True)
    print(f"written {JSON_PATH.relative_to(RESULTS_DIR.parent)}  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
