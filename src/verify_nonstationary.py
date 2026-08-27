"""
P5.1 experiment driver -- GammaTune vs fixed gamma on NON-stationary prompt streams.

Reference: notes/project_plan_v9.md sec 7 P5.1 + sec 9.6 pitfall 9 + sec 13.

The controller state (gamma, gamma_bar) is carried across every prompt in a
sequence via gammatune_generate's carry_state, so gamma never resets at a prompt
boundary -- only the task type changes under it. Fixed-gamma baselines have no
state to carry; each prompt is an independent speculative_generate at that gamma.

Metrics (PRIMARY = mean emitted tokens per round, same rationale as P5.0):
  * whole-sequence mean emitted/round, GammaTune vs each fixed gamma;
  * per-segment mean emitted/round (segment A = code/structured, segment B = open
    chat) -- shows whether the effective step size tracks alpha up in A and down
    in B;
  * post-switch window: the first POST_SWITCH_ROUNDS rounds at/after each task
    switch, to measure how fast GammaTune re-adjusts;
  * the full gamma trajectory (per seed) with prompt boundaries and labels, so the
    "expand fast, contract slow" behaviour can be inspected / plotted.

Then compared against the steady-state P5.0 result (results/p5_0_gammatune.json,
if present): does GammaTune's relative advantage grow, hold, or shrink when the
workload is non-stationary?

>= 3 seeds (0,1,2), sampling mode (temperature = 1.0). Headline numbers are
mean +/- std across seeds; overlapping +/-1 std intervals are called out.

If GammaTune trails a fixed gamma on a sequence, that is recorded as a
reproduction of pitfall 9 / the paper's Limitations, NOT as a failure.

Run:  python src/verify_nonstationary.py [--max-new-tokens N] [--smoke]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gammatune import GammaTuneConfig, gammatune_generate  # noqa: E402
from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer  # noqa: E402
from nonstationary_prompts import SEQUENCES, switch_indices  # noqa: E402
from rejection_sampling import speculative_generate  # noqa: E402
from verify_gammatune import measure_c  # noqa: E402

FIXED_GAMMAS = [1, 3, 5, 7]
SEEDS = [0, 1, 2]
TEMPERATURE = 1.0
POST_SWITCH_ROUNDS = 6
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p5_1_nonstationary.json"
P5_0_PATH = Path(__file__).resolve().parent.parent / "results" / "p5_0_gammatune.json"

CONFIG = GammaTuneConfig()  # paper defaults, frozen (sec 9.6 risk 1)


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    m = statistics.fmean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def _run_sequence_gammatune(sequence, draft, target, tok, seed, max_new_tokens):
    rounds, carry = [], None
    for pi, (label, prompt) in enumerate(sequence):
        r = gammatune_generate(
            prompt, draft, target, tok, config=CONFIG,
            max_new_tokens=max_new_tokens, temperature=TEMPERATURE, seed=seed, carry_state=carry,
        )
        carry = r.final_state
        for k in range(r.n_rounds):
            rounds.append({"pi": pi, "label": label, "emitted": r.emitted_per_round[k],
                           "accept": r.accept_lengths[k], "gamma": r.gamma_trace[k]})
    return rounds


def _run_sequence_fixed(sequence, gamma, draft, target, tok, seed, max_new_tokens):
    rounds = []
    for pi, (label, prompt) in enumerate(sequence):
        g = speculative_generate(
            prompt, draft, target, tok,
            gamma=gamma, max_new_tokens=max_new_tokens, temperature=TEMPERATURE, seed=seed,
        )
        for k in range(g.n_rounds):
            rounds.append({"pi": pi, "label": label, "emitted": g.emitted_per_round[k],
                           "accept": g.accept_lengths[k], "gamma": gamma})
    return rounds


def _seq_metrics(rounds, sequence):
    """Per-seed scalar metrics from one run's round list."""
    emitted = [r["emitted"] for r in rounds]
    by_label = {}
    for lab in ("A", "B"):
        e = [r["emitted"] for r in rounds if r["label"] == lab]
        by_label[lab] = statistics.fmean(e) if e else None
    windows = []
    for si in switch_indices(sequence):
        w = [r for r in rounds if r["pi"] >= si][:POST_SWITCH_ROUNDS]
        if not w:
            continue
        windows.append({
            "switch_prompt_idx": si,
            "to_label": sequence[si][0],
            "mean_emitted": statistics.fmean(r["emitted"] for r in w),
            "gamma_first": w[0]["gamma"],
            "gamma_last": w[-1]["gamma"],
        })
    return {
        "whole_seq_mean_emitted": statistics.fmean(emitted) if emitted else 0.0,
        "seg_A_mean_emitted": by_label["A"],
        "seg_B_mean_emitted": by_label["B"],
        "post_switch_windows": windows,
        "n_rounds": len(rounds),
        "emitted_sum": sum(emitted),
        "gamma_sum": sum(r["gamma"] for r in rounds),
    }


def _aggregate(per_seed_metrics):
    """mean +/- std across seeds for the scalar fields; windows aligned by position."""
    whole = _mean_std(m["whole_seq_mean_emitted"] for m in per_seed_metrics)
    segA = _mean_std(m["seg_A_mean_emitted"] for m in per_seed_metrics if m["seg_A_mean_emitted"] is not None)
    segB = _mean_std(m["seg_B_mean_emitted"] for m in per_seed_metrics if m["seg_B_mean_emitted"] is not None)
    n_windows = min(len(m["post_switch_windows"]) for m in per_seed_metrics)
    windows_agg = []
    for j in range(n_windows):
        ws = [m["post_switch_windows"][j] for m in per_seed_metrics]
        wm, ws_std = _mean_std(w["mean_emitted"] for w in ws)
        windows_agg.append({
            "switch_prompt_idx": ws[0]["switch_prompt_idx"],
            "to_label": ws[0]["to_label"],
            "mean_emitted_mean": wm,
            "mean_emitted_std": ws_std,
            "gamma_first_per_seed": [w["gamma_first"] for w in ws],
            "gamma_last_per_seed": [w["gamma_last"] for w in ws],
        })
    return {
        "whole_seq_mean_emitted": whole[0],
        "whole_seq_mean_emitted_std": whole[1],
        "seg_A_mean_emitted": segA[0],
        "seg_A_mean_emitted_std": segA[1],
        "seg_B_mean_emitted": segB[0],
        "seg_B_mean_emitted_std": segB[1],
        "post_switch_windows": windows_agg,
        "emitted_sum_pooled": sum(m["emitted_sum"] for m in per_seed_metrics),
        "gamma_sum_pooled": sum(m["gamma_sum"] for m in per_seed_metrics),
        "n_rounds_pooled": sum(m["n_rounds"] for m in per_seed_metrics),
    }


def _cost_model_supplement(per_config, measured_c):
    """SUPPLEMENTARY -- see verify_gammatune._cost_model_supplement and sec 9.2
    pitfall 14. whole_seq_mean_emitted is monotone in gamma; this reweights each
    round by the cost-model (c + gamma) factor so the draft-forward waste of a
    large fixed gamma is charged."""
    c_values = {"measured": round(measured_c, 2), "c4": 4.0, "c7": 7.0, "c10": 10.0}
    for name, r in per_config.items():
        r["cost_model_adjusted"] = {
            k: (r["emitted_sum_pooled"] / (r["n_rounds_pooled"] * c + r["gamma_sum_pooled"])
                if (r["n_rounds_pooled"] * c + r["gamma_sum_pooled"]) else 0.0)
            for k, c in c_values.items()
        }
    gt = per_config["gammatune"]
    fixed = {k: v for k, v in per_config.items() if k.startswith("fixed_gamma_")}
    per_c = {}
    for k in c_values:
        best_name = max(fixed, key=lambda n: fixed[n]["cost_model_adjusted"][k])
        gt_v = gt["cost_model_adjusted"][k]
        bf_v = fixed[best_name]["cost_model_adjusted"][k]
        rel = (gt_v / bf_v - 1.0) * 100 if bf_v else 0.0
        per_c[k] = f"c={c_values[k]}: GammaTune {gt_v:.4f} vs best fixed {best_name} {bf_v:.4f} ({rel:+.1f}%)"
    return {"c_values_used": c_values, "per_c": per_c}


def _verdict(seq_name, per_config):
    gt = per_config["gammatune"]
    fixed = {k: v for k, v in per_config.items() if k.startswith("fixed_gamma_")}
    best_name = max(fixed, key=lambda k: fixed[k]["whole_seq_mean_emitted"])
    best = fixed[best_name]
    gt_m, gt_s = gt["whole_seq_mean_emitted"], gt["whole_seq_mean_emitted_std"]
    bf_m, bf_s = best["whole_seq_mean_emitted"], best["whole_seq_mean_emitted_std"]
    overlap = (gt_m + gt_s) >= (bf_m - bf_s) and (bf_m + bf_s) >= (gt_m - gt_s)
    if gt_m >= bf_m:
        v = (f"[{seq_name}] GammaTune {gt_m:.3f} >= best fixed {best_name} {bf_m:.3f}: "
             f"the EMA + fast-expand design holds up at this switch frequency.")
    elif overlap:
        v = (f"[{seq_name}] GammaTune {gt_m:.3f} +/- {gt_s:.3f} vs best fixed {best_name} "
             f"{bf_m:.3f} +/- {bf_s:.3f}: overlapping intervals -- statistical tie.")
    else:
        v = (f"[{seq_name}] GammaTune {gt_m:.3f} +/- {gt_s:.3f} trails best fixed {best_name} "
             f"{bf_m:.3f} +/- {bf_s:.3f} (disjoint). Reproduces pitfall 9 / the paper's "
             f"Limitations: a history-based controller loses ground under this switch rate. "
             f"Not a failure -- it is evidence for needing stronger scene-awareness (P5.3).")
    return {"best_fixed_config": best_name, "within_1std": overlap,
            "gammatune_whole_seq": [gt_m, gt_s], "best_fixed_whole_seq": [bf_m, bf_s],
            "verdict": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    global SEEDS
    sequences = dict(SEQUENCES)
    if args.smoke:
        SEEDS = [0]
        args.max_new_tokens = 16
        sequences = {"A_to_B": SEQUENCES["A_to_B"][:2] + SEQUENCES["A_to_B"][-2:]}

    print(f"draft  = {DRAFT_MODEL_NAME}\ntarget = {TARGET_MODEL_NAME}")
    print(f"sequences = {list(sequences)}, fixed gammas = {FIXED_GAMMAS}, seeds = {SEEDS}, "
          f"temp = {TEMPERATURE}, max_new_tokens = {args.max_new_tokens}")
    print(f"GammaTune config = {CONFIG}\n", flush=True)

    draft_model, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target_model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    c_meas = measure_c(draft_model, target_model, tokenizer)
    print(f"measured c = T_target/T_draft = {c_meas['c']:.2f}\n", flush=True)

    out_sequences = {}
    for seq_name, sequence in sequences.items():
        print(f"==== sequence {seq_name}  ({len(sequence)} prompts, "
              f"switches at {switch_indices(sequence)}) ====", flush=True)
        per_config = {}
        gamma_traces = {}

        for gamma in FIXED_GAMMAS:
            name = f"fixed_gamma_{gamma}"
            per_seed = []
            for seed in SEEDS:
                rounds = _run_sequence_fixed(sequence, gamma, draft_model, target_model,
                                             tokenizer, seed, args.max_new_tokens)
                per_seed.append(_seq_metrics(rounds, sequence))
            per_config[name] = _aggregate(per_seed)
            print(f"  {name:<16} whole-seq emitted/round "
                  f"{per_config[name]['whole_seq_mean_emitted']:.3f} "
                  f"+/- {per_config[name]['whole_seq_mean_emitted_std']:.3f}", flush=True)

        per_seed_gt = []
        for seed in SEEDS:
            rounds = _run_sequence_gammatune(sequence, draft_model, target_model,
                                             tokenizer, seed, args.max_new_tokens)
            per_seed_gt.append(_seq_metrics(rounds, sequence))
            gamma_traces[f"seed_{seed}"] = [
                {"pi": r["pi"], "label": r["label"], "gamma": r["gamma"], "accept": r["accept"]}
                for r in rounds
            ]
        per_config["gammatune"] = _aggregate(per_seed_gt)
        print(f"  {'gammatune':<16} whole-seq emitted/round "
              f"{per_config['gammatune']['whole_seq_mean_emitted']:.3f} "
              f"+/- {per_config['gammatune']['whole_seq_mean_emitted_std']:.3f}", flush=True)
        print(f"  {'gammatune':<16} seg A {per_config['gammatune']['seg_A_mean_emitted']:.3f} | "
              f"seg B {per_config['gammatune']['seg_B_mean_emitted']:.3f}", flush=True)

        verdict = _verdict(seq_name, per_config)
        cost_supp = _cost_model_supplement(per_config, c_meas["c"])
        print(f"  -> {verdict['verdict']}", flush=True)
        print(f"     cost-model supplement: {cost_supp['per_c']['measured']}\n", flush=True)
        out_sequences[seq_name] = {
            "prompt_labels": [lab for lab, _ in sequence],
            "switch_indices": switch_indices(sequence),
            "per_config": per_config,
            "gammatune_gamma_traces": gamma_traces,
            "verdict": verdict,
            "cost_model_supplement": cost_supp,
        }

    # compare with steady-state P5.0
    steady_cmp = None
    if P5_0_PATH.exists():
        p50 = json.loads(P5_0_PATH.read_text())
        p50_rows = {r["config"]: r for r in p50["per_config"]}
        if "gammatune" in p50_rows:
            gt_steady = p50_rows["gammatune"]["mean_emitted_per_round"]
            best_fixed_steady = max(
                (r["mean_emitted_per_round"] for k, r in p50_rows.items()
                 if k.startswith("fixed_gamma_")), default=0.0)
            steady_adv = gt_steady - best_fixed_steady
            per_seq_adv = {}
            for seq_name, s in out_sequences.items():
                gt_ns = s["per_config"]["gammatune"]["whole_seq_mean_emitted"]
                bf_ns = max(v["whole_seq_mean_emitted"] for k, v in s["per_config"].items()
                            if k.startswith("fixed_gamma_"))
                per_seq_adv[seq_name] = {
                    "gammatune_minus_best_fixed": gt_ns - bf_ns,
                    "vs_steady_delta": (gt_ns - bf_ns) - steady_adv,
                }
            steady_cmp = {
                "steady_gammatune_minus_best_fixed": steady_adv,
                "per_sequence": per_seq_adv,
                "note": ("positive vs_steady_delta = GammaTune's edge over the best fixed gamma is "
                         "LARGER in the non-stationary stream than in steady state; negative = smaller."),
            }

    result = {
        "task": "P5.1",
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "max_new_tokens": args.max_new_tokens,
        "post_switch_window_rounds": POST_SWITCH_ROUNDS,
        "gammatune_config": vars(CONFIG),
        "primary_metric": "mean_emitted_per_round = mean(n_accepted)+1 = tokens per target forward",
        "wall_clock_caveat": "MPS, no KV cache -- wall-clock not reported here; see P4/cloud.",
        "measured_c": c_meas,
        "sequences": out_sequences,
        "steady_state_comparison": steady_cmp,
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))
    print("=" * 68)
    for seq_name, s in out_sequences.items():
        print(f"{seq_name}: {s['verdict']['verdict']}")
    print(f"\nwritten to {RESULTS_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
