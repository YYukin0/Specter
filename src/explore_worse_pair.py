"""
Side experiment (NOT in the plan's P-numbering) -- does adaptivity's payoff depend
on acceptance-rate variance?

Motivation: P5.0 / P5.1 found GammaTune does not beat fixed gamma on the mainline
pair (draft = Qwen2.5-0.5B-Instruct, target = Qwen2.5-1.5B-Instruct). The
attribution (pitfall 9) was "alpha ~= 0.79 is too stable and accept-length
variance is too low for an adaptive controller to have anything to exploit".
This script tests that attribution directly by swapping in a DELIBERATELY
worse-matched pair and re-running the same comparison.

The mainline pair is untouched -- M2/M4/M5 still use 0.5B-Instruct / 1.5B-Instruct.
All output here is prefixed `explore_` and never overwrites p5_0 / p5_1.

Explore pair 1 (default): draft = Qwen/Qwen2.5-0.5B  (BASE, not instruct)
                          target = Qwen/Qwen2.5-1.5B-Instruct (unchanged)
  Same tokenizer (pitfall 1 satisfied), but a base draft does not follow the
  instruct target's style/format, so its next-token distribution should diverge
  more -> lower alpha, larger accept-length variance.
Explore pair 2 (--pair2): draft = Qwen/Qwen2.5-0.5B-Instruct
                          target = Qwen/Qwen2.5-3B-Instruct  (capability gap 6x)
  Use this if pair 1's alpha does not drop clearly below the mainline.

Two phases, one model load:
  Phase 1 -- alpha + accept-length variance sweep. fixed gamma in {1,3,5,7},
    seeds {0,1,2}, 8 prompts, sampling temp=1.0. Reports alpha (mean +/- std
    across seeds) and pooled accept_lengths mean / std / histogram per gamma,
    next to the mainline P1.4 reference. -> results/explore_worse_pair_<pairN>_alpha.json
  Phase 2 -- GammaTune vs fixed gamma in this regime, reusing the P5.0 driver
    functions (measure_c, _run_fixed, _run_gammatune, _verdict,
    _cost_model_supplement). Primary metric = mean_emitted_per_round + the
    cost-model supplement, same as P5.0. -> results/explore_worse_pair_<pairN>_gammatune.json

Run:  python src/explore_worse_pair.py [--pair2] [--max-new-tokens N] [--smoke]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_gammatune as vg  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402
from rejection_sampling import speculative_generate  # noqa: E402

PAIR1 = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B-Instruct")
PAIR2 = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
MAINLINE = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")

# mainline reference (notes/project_plan_v9.md + TASKS.md): P1.4 accept-length std
# by gamma, and the stable alpha the pitfall-9 attribution rests on.
MAINLINE_REF = {
    "alpha": 0.79,
    "accept_len_std_by_gamma": {"1": 0.42, "3": 1.24, "5": 1.95, "7": 2.55, "10": 3.24},
    "p5_0_verdict": ("GammaTune mean_emitted_per_round 2.94 vs best fixed gamma=7 3.67 "
                     "(disjoint +/-1 std) -- GammaTune lost by ~20% on the primary metric; "
                     "cost-model supplement put it within the optimal cluster (0.8-5.9%)."),
}

RES_DIR = Path(__file__).resolve().parent.parent / "results"
SWEEP_GAMMAS = [1, 3, 5, 7]
SEEDS = [0, 1, 2]
TEMPERATURE = 1.0


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


# --------------------------------------------------------------------------- #
# Phase 1: alpha + accept-length variance
# --------------------------------------------------------------------------- #
def phase1_alpha_variance(draft, target, tokenizer, prompts, max_new_tokens):
    rows = []
    for gamma in SWEEP_GAMMAS:
        per_seed_alpha = []
        accept_pool = []
        for seed in SEEDS:
            seed_alpha = []
            for prompt in prompts:
                g = speculative_generate(
                    prompt, draft, target, tokenizer,
                    gamma=gamma, max_new_tokens=max_new_tokens,
                    temperature=TEMPERATURE, seed=seed,
                )
                seed_alpha.append(g.alpha)
                accept_pool.extend(g.accept_lengths)
            per_seed_alpha.append(statistics.fmean(seed_alpha))
        a_m, a_s = _mean_std(per_seed_alpha)
        ac_m, ac_s = _mean_std(accept_pool)
        rows.append({
            "gamma": gamma,
            "alpha_mean": a_m,
            "alpha_std_across_seeds": a_s,
            "accept_len_mean": ac_m,
            "accept_len_pooled_std": ac_s,
            "accept_len_hist": {str(k): accept_pool.count(k) for k in range(gamma + 1)},
            "n_rounds_pooled": len(accept_pool),
            "mainline_accept_len_std": MAINLINE_REF["accept_len_std_by_gamma"].get(str(gamma)),
        })
        print(f"  gamma={gamma}: alpha {a_m:.3f}+/-{a_s:.3f}  accept_len {ac_m:.2f} "
              f"std {ac_s:.2f} (mainline std {MAINLINE_REF['accept_len_std_by_gamma'].get(str(gamma))})",
              flush=True)
    return rows


def _phase1_read(rows):
    g3 = next(r for r in rows if r["gamma"] == 3)
    g5 = next(r for r in rows if r["gamma"] == 5)
    alpha_lower = g3["alpha_mean"] < 0.70
    var_bigger = (g3["accept_len_pooled_std"] > 1.4 * MAINLINE_REF["accept_len_std_by_gamma"]["3"]
                  or g5["accept_len_pooled_std"] > 1.4 * MAINLINE_REF["accept_len_std_by_gamma"]["5"])
    if alpha_lower and var_bigger:
        rec = ("GOOD REGIME: alpha at gamma=3 dropped below 0.70 and accept-length variance is "
               ">1.4x the mainline. This is the higher-variance regime the experiment wants; "
               "phase 2 (GammaTune vs fixed) on this pair is meaningful.")
        switch = False
    elif alpha_lower or var_bigger:
        rec = ("PARTIAL: only one of {alpha clearly lower, variance clearly larger} holds. "
               "Phase 2 is still run on this pair, but consider re-running with --pair2 "
               "(0.5B-Instruct + 3B-Instruct) for a cleaner high-variance regime.")
        switch = True
    else:
        rec = ("WEAK: alpha is still >= 0.70 and variance is not clearly larger than mainline. "
               "This pair is not meaningfully worse-matched; re-run with --pair2.")
        switch = True
    return {"alpha_g3": g3["alpha_mean"], "accept_len_std_g3": g3["accept_len_pooled_std"],
            "accept_len_std_g5": g5["accept_len_pooled_std"],
            "mainline_alpha": MAINLINE_REF["alpha"], "recommend_switch_to_pair2": switch,
            "reading": rec}


# --------------------------------------------------------------------------- #
# Phase 2: GammaTune vs fixed gamma (reuse the P5.0 driver)
# --------------------------------------------------------------------------- #
def phase2_gammatune(draft, target, tokenizer, max_new_tokens):
    vg.SEEDS = SEEDS
    print("  -- measuring c --", flush=True)
    c_meas = vg.measure_c(draft, target, tokenizer)
    print(f"     c = {c_meas['c']:.2f}", flush=True)
    base = vg._target_only_tps(target, tokenizer, max_new_tokens)
    rows = []
    for gamma in vg.FIXED_GAMMAS:
        print(f"  -- fixed gamma = {gamma} --", flush=True)
        rows.append(vg._run_fixed(gamma, draft, target, tokenizer,
                                  base["tok_per_s_mean"], max_new_tokens))
    print("  -- GammaTune --", flush=True)
    rows.append(vg._run_gammatune(draft, target, tokenizer,
                                  base["tok_per_s_mean"], max_new_tokens))
    verdict = vg._verdict(rows)
    cost_supp = vg._cost_model_supplement(rows, c_meas["c"])
    return {"measured_c": c_meas, "target_only_baseline": base,
            "per_config": rows, "verdict": verdict, "cost_model_supplement": cost_supp}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair2", action="store_true",
                    help="use draft=0.5B-Instruct, target=3B-Instruct instead of the default "
                         "base-0.5B / 1.5B-Instruct pair")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 prompts, 16 tokens")
    args = ap.parse_args()

    global SEEDS
    draft_name, target_name = (PAIR2 if args.pair2 else PAIR1)
    tag = "pair2_0.5Bi_3Bi" if args.pair2 else "pair1_base0.5B_1.5Bi"

    prompts = list(vg.PROMPTS)
    if args.smoke:
        SEEDS = [0]
        prompts = prompts[:2]
        vg.PROMPTS[:] = prompts
        args.max_new_tokens = 16

    print(f"SIDE EXPERIMENT -- worse-matched pair ({tag})")
    print(f"draft  = {draft_name}")
    print(f"target = {target_name}")
    print(f"mainline (untouched) = {MAINLINE}")
    print(f"gammas = {SWEEP_GAMMAS} (phase 1) / {vg.FIXED_GAMMAS} (phase 2), seeds = {SEEDS}, "
          f"temp = {TEMPERATURE}, max_new_tokens = {args.max_new_tokens}\n", flush=True)

    draft, _ = load_model_and_tokenizer(draft_name)
    target, tokenizer = load_model_and_tokenizer(target_name)

    print("== PHASE 1: alpha + accept-length variance ==", flush=True)
    p1_rows = phase1_alpha_variance(draft, target, tokenizer, prompts, args.max_new_tokens)
    p1_read = _phase1_read(p1_rows)
    print(f"\n  {p1_read['reading']}\n", flush=True)

    alpha_out = {
        "experiment": "side experiment: acceptance-rate variance vs adaptivity payoff (not a plan P-number)",
        "pair_tag": tag,
        "draft_model": draft_name,
        "target_model": target_name,
        "mainline_pair": list(MAINLINE),
        "mainline_reference": MAINLINE_REF,
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "max_new_tokens": args.max_new_tokens,
        "n_prompts": len(prompts),
        "per_gamma": p1_rows,
        "reading": p1_read,
    }
    alpha_path = RES_DIR / f"explore_worse_pair_{tag.split('_')[0]}_alpha.json"
    alpha_path.write_text(json.dumps(alpha_out, indent=2))
    print(f"  written {alpha_path.name}\n", flush=True)

    print("== PHASE 2: GammaTune vs fixed gamma in this regime ==", flush=True)
    p2 = phase2_gammatune(draft, target, tokenizer, args.max_new_tokens)

    gt = next(r for r in p2["per_config"] if r["config"] == "gammatune")
    fixed = [r for r in p2["per_config"] if r["config"].startswith("fixed_gamma_")]
    best = max(fixed, key=lambda r: r["mean_emitted_per_round"])
    gt_m = gt["mean_emitted_per_round"]
    bf_m = best["mean_emitted_per_round"]
    rel = (gt_m / bf_m - 1.0) * 100 if bf_m else 0.0

    conclusion = (
        f"PAIR {tag}: alpha(gamma=3) = {p1_read['alpha_g3']:.3f} vs mainline ~{MAINLINE_REF['alpha']:.2f}; "
        f"accept-length std gamma=3/5 = {p1_read['accept_len_std_g3']:.2f}/{p1_read['accept_len_std_g5']:.2f} "
        f"vs mainline {MAINLINE_REF['accept_len_std_by_gamma']['3']}/{MAINLINE_REF['accept_len_std_by_gamma']['5']}. "
        f"Phase 2 primary metric: GammaTune {gt_m:.3f} vs best fixed {best['config']} {bf_m:.3f} "
        f"({rel:+.1f}%). "
    )
    if rel >= -1.0:
        conclusion += ("GammaTune now MATCHES OR BEATS the best fixed gamma on the primary metric -- "
                       "this supports the causal claim that adaptivity's payoff scales with "
                       "acceptance-rate variance: the P5.0 null becomes 'GammaTune is useless in "
                       "the low-variance mainline regime, useful once variance is high'.")
    else:
        conclusion += (f"GammaTune STILL loses on the primary metric ({rel:+.1f}%). Even with lower "
                       "alpha / higher variance here, GammaTune's EMA+expand design does not convert "
                       "that into a win on this pair -- check the cost-model supplement and the gamma "
                       "trace; the variance may lack exploitable structure (base-draft alpha swings "
                       "randomly, EMA cannot track), which still weakens GammaTune's practical claim.")

    gt_out = {
        "experiment": "side experiment phase 2: GammaTune vs fixed gamma on a worse-matched pair",
        "pair_tag": tag,
        "draft_model": draft_name,
        "target_model": target_name,
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "max_new_tokens": args.max_new_tokens,
        "gammatune_config": vars(vg.CONFIG),
        "primary_metric": "mean_emitted_per_round (same as P5.0)",
        "phase1_alpha_summary": p1_read,
        "mainline_reference": MAINLINE_REF,
        **p2,
        "conclusion": conclusion,
    }
    gt_path = RES_DIR / f"explore_worse_pair_{tag.split('_')[0]}_gammatune.json"
    gt_path.write_text(json.dumps(gt_out, indent=2))

    print("\n" + "=" * 72)
    print(conclusion)
    print(f"\nwritten {alpha_path.name} + {gt_path.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
