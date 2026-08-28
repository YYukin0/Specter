"""
P1.4 -- gamma sweep for the P1.1 speculative decoder.

gamma in {1, 3, 5, 7, 10, 16} (notes/project_plan_v9.md sec 7 P1.4 + sec 9.6 risk 2).
gamma=16 was added 2026-08-28 so the accept-length-std curve has a point opposite
every one of the three AdaEDL Fig 7c reference values the plan cites (DL=3/7/16 ->
std ~= 1.2 / 1.92 / 2.35). See adaedl_fig7c_comparison in the result file; the
magnitudes are not expected to match (different model pair / dataset), only the
monotone-increasing shape.

What is recorded, and why:
  * The full distribution of per-round accept lengths (n_accepted, 0..gamma), not
    just the mean -- the plan cites AdaEDL's result that accept-length variance
    grows with gamma as the empirical basis for "why pillar 5 needs an adaptive
    gamma". So mean AND standard deviation are both reported per gamma.
  * >= 3 runs per gamma (seeds 0,1,2), sampling mode (temperature = 1.0) so the
    repeats genuinely differ and the reported std is real. Headline numbers are
    mean +/- std across the 3 seeds; if a std is large enough that a claim like
    "gamma=5 beats gamma=3" has overlapping intervals, that is stated outright
    rather than hidden behind the mean (risk 2).

Throughput caveat: these wall-clock speedups are indicative only. Neither the
speculative path nor the target-only baseline here uses a KV cache (P1.x
prioritises obvious correctness), and this runs on MPS. The real throughput /
batch curve is P4. The accept-length distribution and its variance, which are the
actual P1.4 deliverable, do not depend on timing.

Run:  python src/gamma_sweep.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from prompts import PROMPTS
from rejection_sampling import speculative_generate, target_only_generate

GAMMAS = [1, 3, 5, 7, 10, 16]

# AdaEDL paper Fig 7c (Dolly-15k, Llama2-7B target, aligned 115M draft): the three
# accept-length-std points the plan quotes as the empirical basis for "pillar 5
# needs an adaptive gamma". Direction-only reference -- different model pair and
# dataset, so absolute magnitude is not expected to line up.
ADAEDL_FIG7C_STD_REFERENCE = {3: 1.2, 7: 1.92, 16: 2.35}
SEEDS = [0, 1, 2]
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 48
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p1_4_gamma_sweep.json"


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def target_only_baseline(target_model, tokenizer):
    """tok/s for plain target-only sampling, per seed (mean over prompts)."""
    per_seed = []
    for seed in SEEDS:
        toks = t = 0.0
        for prompt in PROMPTS:
            r = target_only_generate(
                prompt, target_model, tokenizer,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, seed=seed,
            )
            toks += len(r.token_ids)
            t += r.elapsed_s
        per_seed.append(toks / t if t else 0.0)
    m, s = _mean_std(per_seed)
    return {"tok_per_s_mean": m, "tok_per_s_std": s, "per_seed": per_seed}


def sweep_one_gamma(gamma, draft_model, target_model, tokenizer, baseline_tps):
    accept_lengths_all = []      # pooled n_accepted over every round, every prompt, every seed
    emitted_all = []
    per_seed_tps = []
    per_seed_speedup = []
    per_seed_mean_accept = []
    for seed in SEEDS:
        toks = t = 0.0
        seed_accepts = []
        for prompt in PROMPTS:
            g = speculative_generate(
                prompt, draft_model, target_model, tokenizer,
                gamma=gamma, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, seed=seed,
            )
            accept_lengths_all.extend(g.accept_lengths)
            emitted_all.extend(g.emitted_per_round)
            seed_accepts.extend(g.accept_lengths)
            toks += len(g.token_ids)
            t += g.elapsed_s
        tps = toks / t if t else 0.0
        per_seed_tps.append(tps)
        per_seed_speedup.append(tps / baseline_tps if baseline_tps else 0.0)
        per_seed_mean_accept.append(statistics.fmean(seed_accepts) if seed_accepts else 0.0)

    al_mean, al_std = _mean_std(accept_lengths_all)
    em_mean, em_std = _mean_std(emitted_all)
    sp_mean, sp_std = _mean_std(per_seed_speedup)
    hist = {k: accept_lengths_all.count(k) for k in range(gamma + 1)}
    return {
        "gamma": gamma,
        "n_rounds_pooled": len(accept_lengths_all),
        "accept_length_mean": al_mean,          # n_accepted (drafts accepted / round)
        "accept_length_std": al_std,
        "accept_length_hist": hist,
        "emitted_per_round_mean": em_mean,       # n_accepted + 1 (tokens committed / round)
        "emitted_per_round_std": em_std,
        "per_seed_mean_accept_length": per_seed_mean_accept,
        "tok_per_s_per_seed": per_seed_tps,
        "speedup_vs_target_only_mean": sp_mean,
        "speedup_vs_target_only_std": sp_std,
    }


def significance_notes(rows):
    """Flag adjacent-gamma speedup comparisons whose mean +/- std intervals overlap."""
    notes = []
    for a, b in zip(rows, rows[1:]):
        ai = (a["speedup_vs_target_only_mean"] - a["speedup_vs_target_only_std"],
              a["speedup_vs_target_only_mean"] + a["speedup_vs_target_only_std"])
        bi = (b["speedup_vs_target_only_mean"] - b["speedup_vs_target_only_std"],
              b["speedup_vs_target_only_mean"] + b["speedup_vs_target_only_std"])
        overlap = not (ai[1] < bi[0] or bi[1] < ai[0])
        better = "gamma=%d" % (b["gamma"] if b["speedup_vs_target_only_mean"] > a["speedup_vs_target_only_mean"] else a["gamma"])
        notes.append({
            "pair": f"gamma={a['gamma']} vs gamma={b['gamma']}",
            "speedup_mean": [a["speedup_vs_target_only_mean"], b["speedup_vs_target_only_mean"]],
            "intervals_overlap": overlap,
            "verdict": (f"{better} numerically faster but +/-1 std intervals overlap -- "
                        "not statistically significant at this run count")
            if overlap else f"{better} faster, intervals disjoint",
        })
    return notes


def main():
    print(f"draft  = {DRAFT_MODEL_NAME}")
    print(f"target = {TARGET_MODEL_NAME}")
    print(f"gammas = {GAMMAS}, seeds = {SEEDS}, temperature = {TEMPERATURE}, "
          f"max_new_tokens = {MAX_NEW_TOKENS}\n")

    draft_model, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target_model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    print("-- target-only baseline --")
    base = target_only_baseline(target_model, tokenizer)
    print(f"   {base['tok_per_s_mean']:.2f} +/- {base['tok_per_s_std']:.2f} tok/s\n")

    rows = []
    for gamma in GAMMAS:
        print(f"-- gamma = {gamma} --")
        row = sweep_one_gamma(gamma, draft_model, target_model, tokenizer, base["tok_per_s_mean"])
        rows.append(row)
        print(f"   accept_len {row['accept_length_mean']:.2f} +/- {row['accept_length_std']:.2f}  "
              f"hist={row['accept_length_hist']}")
        print(f"   speedup {row['speedup_vs_target_only_mean']:.2f}x +/- {row['speedup_vs_target_only_std']:.2f}\n")

    notes = significance_notes(rows)

    std_by_gamma = {r["gamma"]: r["accept_length_std"] for r in rows}
    adaedl_cmp = {
        "reference": "AdaEDL Fig 7c (Dolly-15k, Llama2-7B target, aligned 115M draft) -- direction only",
        "note": ("our draft/target pair and eval prompts differ from AdaEDL's, so absolute "
                 "std magnitude is not expected to match; the test is whether std rises "
                 "monotonically with gamma the way their figure shows"),
        "points": [
            {"gamma": g, "ours_accept_length_std": std_by_gamma.get(g),
             "adaedl_fig7c_std": ref}
            for g, ref in sorted(ADAEDL_FIG7C_STD_REFERENCE.items())
        ],
        "ours_monotone_increasing_in_gamma": all(
            a["accept_length_std"] <= b["accept_length_std"] + 1e-9
            for a, b in zip(rows, rows[1:])
        ),
    }

    result = {
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "target_only_baseline": base,
        "per_gamma": rows,
        "variance_grows_with_gamma": [
            {"gamma": r["gamma"], "accept_length_std": r["accept_length_std"]} for r in rows
        ],
        "adaedl_fig7c_comparison": adaedl_cmp,
        "adjacent_significance_notes": notes,
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print("=" * 60)
    print("gamma :  accept_len(mean+/-std) :  speedup(mean+/-std)")
    for r in rows:
        print(f"  {r['gamma']:>2}  :  {r['accept_length_mean']:.2f} +/- {r['accept_length_std']:.2f}"
              f"          :  {r['speedup_vs_target_only_mean']:.2f}x +/- {r['speedup_vs_target_only_std']:.2f}")
    print(f"\nwritten to {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
