"""
P5.3 experiment driver -- always-spec vs always-target vs batch-aware circuit
breaker, on a synthetic batch-size signal.

Reference: notes/project_plan_v9.md sec 7 P5.3 + sec 9.2 pitfalls 7 / 11 / 12 / 14
+ sec 9.6 risks 1 / 2 / 4 + notes/M4-P5.3-Batch熔断器任务prompt_2026-08-28.md.

The batch-size signal is SYNTHETIC and injected -- there is no real high-batch
throughput on this machine (that is M5[A] / cloud). Two traces:
  * single spike : low -> high -> low
  * double spike : low -> high -> low -> high -> low
Each trace value is one generation round (a decision point). Segment lengths are
documented in `TRACES` below.

Metrics (pitfall 14 -- NOT mean_emitted_per_round, which has no draft-cost term):

  Metric A  (task decision point 6, as written): cost-model throughput
      total_emitted / total_cost_units
      speculative / probe round costs (c + gamma); degraded target step costs c.
    NOTE this metric has *no batch-dependent term*: a speculative round costs the
    same whether the batch is 1 or 1000. So on Metric A always-spec is an upper
    bound the breaker can only tie -- degrading strictly does less useful work
    per unit compute. This is pitfall 14 resurfacing (a metric with no term for
    the thing the mechanism trades away cannot reward the mechanism). Reported as
    the headline anyway, per sec 9.6 risk 1 (do not redefine the metric to win).

  Metric B  (saturation-aware sensitivity, clearly supplementary): under a
    saturated accelerator the draft forwards of a speculative round no longer
    overlap idle capacity -- they contend with the target batch (pitfall 7,
    Nightjar's up-to-30.25% regression). Model that as: while batch >= threshold
    a speculative/probe round's gamma draft forwards cost `sat_tax` cost-units
    each instead of 1, i.e. round cost = c + gamma * sat_tax. Reported for
    sat_tax in {1, 2, 3}; sat_tax ~= 3 reproduces Nightjar's regression order of
    magnitude for c in {4, 7, 10}. On Metric B the breaker's value (skip the tax
    during high-batch segments) becomes visible.

  Secondary: degrade / restore lag vs the batch signal (rounds); re-probe hit
    accuracy (|probe alpha - pooled always-spec alpha|); measured switch-cost
    proxy (ms, pitfall 12) and its share of wall time.
  Reference only: wall-clock tok/s, with the standing MPS/no-KV-cache caveat.

Runs: 8 prompts (src/prompts.py), sampling temperature = 1.0, seeds {0, 1, 2}.
Headline numbers are mean +/- std across seeds; overlapping +/-1 std intervals
are called a tie outright (sec 9.6 risk 2).

Run:  python src/verify_circuit_breaker.py [--max-new-tokens N] [--smoke]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit_breaker import (  # noqa: E402
    CircuitBreakerConfig,
    circuit_breaker_generate,
)
from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer  # noqa: E402
from prompts import PROMPTS  # noqa: E402
from rejection_sampling import speculative_generate, target_only_generate  # noqa: E402
from verify_gammatune import measure_c  # noqa: E402

SEEDS = [0, 1, 2]
TEMPERATURE = 1.0
CONFIG = CircuitBreakerConfig()  # frozen defaults, rationale in circuit_breaker.py
SAT_TAX_VALUES = [1, 2, 3]
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p5_3_circuit_breaker.json"

# Synthetic batch-size traces. One value per generation round. `lo` = 1 (single
# stream), `hi` = 32 (well above CONFIG.batch_threshold=8). Segment lengths chosen
# so the low-batch segments dominate (the regime where speculation should be on)
# with a couple of sustained high-batch spikes long enough to matter.
_LO, _HI = 1, 32
TRACES = {
    # 30 low, 25 high, 60 low  (115 rounds; low-batch fraction ~0.78)
    "single_spike": [_LO] * 30 + [_HI] * 25 + [_LO] * 60,
    # 20 low, 20 high, 25 low, 20 high, 55 low  (140 rounds; low fraction ~0.71)
    "double_spike": [_LO] * 20 + [_HI] * 20 + [_LO] * 25 + [_HI] * 20 + [_LO] * 55,
}


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


# --------------------------------------------------------------------------- #
# Per-round ledgers for the two baselines (so Metric B can see the batch too)
# --------------------------------------------------------------------------- #
def _walk_always_spec(prompts, draft, target, tok, trace, max_new_tokens, seed):
    """Fixed-gamma speculative decoding over the whole stream, no breaker.
    Rounds are mapped onto `trace` by a global round counter."""
    rounds = []
    emitted_total = 0
    acc_num = acc_den = 0
    wall = 0.0
    gstep = 0
    for prompt in prompts:
        g = speculative_generate(
            prompt, draft, target, tok,
            gamma=CONFIG.spec_gamma, max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE, seed=seed,
        )
        wall += g.elapsed_s
        acc_num += g.accepted_total
        acc_den += g.evaluated_total
        for emitted in g.emitted_per_round:
            bs = trace[min(gstep, len(trace) - 1)]
            rounds.append({"mode": "spec", "gamma": CONFIG.spec_gamma, "batch": bs, "emitted": emitted})
            emitted_total += emitted
            gstep += 1
    alpha = acc_num / acc_den if acc_den else 0.0
    return {"rounds": rounds, "emitted_total": emitted_total, "alpha": alpha, "wall_s": wall}


def _walk_always_target(prompts, target, tok, trace, max_new_tokens, seed):
    rounds = []
    emitted_total = 0
    wall = 0.0
    gstep = 0
    for prompt in prompts:
        r = target_only_generate(
            prompt, target, tok,
            max_new_tokens=max_new_tokens, temperature=TEMPERATURE, seed=seed,
        )
        wall += r.elapsed_s
        for _ in r.token_ids:
            bs = trace[min(gstep, len(trace) - 1)]
            rounds.append({"mode": "target", "gamma": 0, "batch": bs, "emitted": 1})
            emitted_total += 1
            gstep += 1
    return {"rounds": rounds, "emitted_total": emitted_total, "alpha": None, "wall_s": wall}


def _cb_rounds(res):
    """Circuit-breaker result -> the same per-round ledger shape."""
    rounds = []
    for mode, gamma, bs, emitted in zip(
        res.per_round_mode, res.per_round_gamma, res.per_round_batch, res.per_round_emitted
    ):
        rounds.append({"mode": mode, "gamma": gamma, "batch": bs, "emitted": emitted})
    return rounds


# --------------------------------------------------------------------------- #
# Cost models
# --------------------------------------------------------------------------- #
def _throughput(rounds, emitted_total, c, sat_tax, batch_threshold):
    """total_emitted / total_cost_units.
      target step            : c
      spec/probe, batch <  T : c + gamma
      spec/probe, batch >= T : c + gamma * sat_tax   (sat_tax=1 -> Metric A)
    """
    cost = 0.0
    for rd in rounds:
        if rd["mode"] == "target":
            cost += c
        else:
            tax = sat_tax if rd["batch"] >= batch_threshold else 1
            cost += c + rd["gamma"] * tax
    return emitted_total / cost if cost else 0.0


# --------------------------------------------------------------------------- #
# Secondary metrics for the circuit-breaker run
# --------------------------------------------------------------------------- #
def _threshold_crossings(trace, threshold):
    """(step, direction) for each round where the batch signal crosses the
    threshold relative to the previous round. direction 'up' = entered high."""
    out = []
    prev_high = False
    for i, bs in enumerate(trace):
        high = bs >= threshold
        if high != prev_high:
            out.append((i, "up" if high else "down"))
        prev_high = high
    return out


def _lag_stats(res, trace, threshold):
    """For each batch-signal crossing, the gap (in rounds) to the next actual
    mode switch in the same direction."""
    crossings = _threshold_crossings(trace[: len(res.per_round_mode)], threshold)
    switches = [(s["step"], s["direction"]) for s in res.mode_switches]
    want = {"up": "spec->target", "down": "target->spec"}
    lags = {"up": [], "down": []}
    for cstep, cdir in crossings:
        after = [ss for ss, sd in switches if sd == want[cdir] and ss >= cstep]
        if after:
            lags[cdir].append(min(after) - cstep)
    return {
        "degrade_lag_rounds": lags["up"],
        "restore_lag_rounds": lags["down"],
        "n_crossings": len(crossings),
        "n_switches": len(switches),
    }


def _reprobe_accuracy(res, reference_alpha):
    if not res.reprobe_log:
        return {"n_reprobes": 0, "mean_abs_dev_from_ref_alpha": None, "reference_alpha": reference_alpha}
    devs = [abs(e["alpha"] - reference_alpha) for e in res.reprobe_log]
    return {
        "n_reprobes": len(res.reprobe_log),
        "reprobe_alpha_mean": statistics.fmean(e["alpha"] for e in res.reprobe_log),
        "reference_alpha": reference_alpha,
        "mean_abs_dev_from_ref_alpha": statistics.fmean(devs),
        "max_abs_dev": max(devs),
    }


# --------------------------------------------------------------------------- #
def run_one_trace(name, trace, draft, target, tok, c_meas, max_new_tokens):
    per_seed = {"always_spec": [], "always_target": [], "circuit_breaker": []}
    cb_secondary = []
    spec_alpha_pool = []
    wall = {"always_spec": [], "always_target": [], "circuit_breaker": []}

    for seed in SEEDS:
        sp = _walk_always_spec(PROMPTS, draft, target, tok, trace, max_new_tokens, seed)
        tg = _walk_always_target(PROMPTS, target, tok, trace, max_new_tokens, seed)
        cb = circuit_breaker_generate(
            PROMPTS, draft, target, tok,
            config=CONFIG, batch_size_trace=trace,
            max_new_tokens=max_new_tokens, temperature=TEMPERATURE, seed=seed,
            measured_c=c_meas,
        )
        spec_alpha_pool.append(sp["alpha"])
        wall["always_spec"].append(sp["wall_s"])
        wall["always_target"].append(tg["wall_s"])
        wall["circuit_breaker"].append(cb.elapsed_s)

        ledgers = {
            "always_spec": (sp["rounds"], sp["emitted_total"]),
            "always_target": (tg["rounds"], tg["emitted_total"]),
            "circuit_breaker": (_cb_rounds(cb), cb.emitted_total),
        }
        c_values = {"measured": c_meas, "c4": 4.0, "c7": 7.0, "c10": 10.0}
        for cfg_name, (rounds, emitted_total) in ledgers.items():
            row = {}
            for tax in SAT_TAX_VALUES:
                row[f"sat_tax_{tax}"] = {
                    ck: _throughput(rounds, emitted_total, cv, tax, CONFIG.batch_threshold)
                    for ck, cv in c_values.items()
                }
            per_seed[cfg_name].append(row)

        cb_secondary.append({
            "seed": seed,
            "lag": _lag_stats(cb, trace, CONFIG.batch_threshold),
            "reprobe": _reprobe_accuracy(cb, sp["alpha"]),
            "n_spec_rounds": cb.spec_rounds,
            "n_target_rounds": cb.target_rounds,
            "n_reprobe_rounds": cb.reprobe_rounds,
            "switch_cost_probe": cb.switch_cost_probe,
            "switch_cost_share_of_wall": (
                len(cb.mode_switches) * cb.switch_cost_probe.get("switch_cost_ms", 0.0) / 1e3
                / cb.elapsed_s if cb.elapsed_s else 0.0
            ),
        })

    # aggregate throughput mean/std across seeds, per (config, sat_tax, c)
    agg = {}
    for cfg_name, seed_rows in per_seed.items():
        agg[cfg_name] = {}
        for tax in SAT_TAX_VALUES:
            key = f"sat_tax_{tax}"
            agg[cfg_name][key] = {}
            for ck in ("measured", "c4", "c7", "c10"):
                vals = [r[key][ck] for r in seed_rows]
                m, s = _mean_std(vals)
                agg[cfg_name][key][ck] = {"mean": m, "std": s, "per_seed": vals}

    verdicts = _verdicts(agg)
    return {
        "trace_name": name,
        "trace_len_rounds": len(trace),
        "trace_segments": _describe_trace(trace, CONFIG.batch_threshold),
        "low_batch_round_fraction": sum(1 for b in trace if b < CONFIG.batch_threshold) / len(trace),
        "throughput": agg,
        "verdicts": verdicts,
        "circuit_breaker_secondary": cb_secondary,
        "wall_s_mean": {k: _mean_std(v)[0] for k, v in wall.items()},
        "pooled_always_spec_alpha": _mean_std(spec_alpha_pool)[0],
    }


def _describe_trace(trace, threshold):
    segs = []
    cur = None
    for b in trace:
        lab = "high" if b >= threshold else "low"
        if cur and cur[0] == lab:
            cur[1] += 1
        else:
            if cur:
                segs.append({"segment": cur[0], "rounds": cur[1]})
            cur = [lab, 1]
    if cur:
        segs.append({"segment": cur[0], "rounds": cur[1]})
    return segs


def _verdicts(agg):
    """Circuit breaker vs max(always_spec, always_target), per sat_tax at c7
    (the representative literature c). Tie = overlapping +/-1 std."""
    out = {}
    for tax in SAT_TAX_VALUES:
        key = f"sat_tax_{tax}"
        cb = agg["circuit_breaker"][key]["c7"]
        sp = agg["always_spec"][key]["c7"]
        tg = agg["always_target"][key]["c7"]
        best_name, best = max([("always_spec", sp), ("always_target", tg)], key=lambda kv: kv[1]["mean"])
        cb_m, cb_s = cb["mean"], cb["std"]
        bf_m, bf_s = best["mean"], best["std"]
        tie = (cb_m + cb_s) >= (bf_m - bf_s) and (bf_m + bf_s) >= (cb_m - cb_s)
        if cb_m >= bf_m:
            v = f"circuit breaker {cb_m:.4f} >= best baseline ({best_name}) {bf_m:.4f} -- meets criterion"
        elif tie:
            v = (f"circuit breaker {cb_m:.4f}+/-{cb_s:.4f} vs {best_name} {bf_m:.4f}+/-{bf_s:.4f}: "
                 f"numerically lower but +/-1 std intervals overlap -- statistical tie, counts")
        else:
            v = (f"circuit breaker {cb_m:.4f}+/-{cb_s:.4f} BELOW {best_name} {bf_m:.4f}+/-{bf_s:.4f} "
                 f"(disjoint). Does not beat the baseline at sat_tax={tax}.")
        out[key] = {
            "c": "c7", "sat_tax": tax,
            "circuit_breaker_mean_std": [cb_m, cb_s],
            "best_baseline": best_name,
            "best_baseline_mean_std": [bf_m, bf_s],
            "tie_within_1std": tie,
            "verdict": v,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 2 prompts, 16 tokens, tiny traces -- pipeline check only")
    args = ap.parse_args()

    global SEEDS, TRACES
    if args.smoke:
        SEEDS = [0]
        PROMPTS[:] = PROMPTS[:2]
        args.max_new_tokens = 16
        TRACES = {
            "single_spike": [_LO] * 4 + [_HI] * 4 + [_LO] * 12,
            "double_spike": [_LO] * 3 + [_HI] * 3 + [_LO] * 4 + [_HI] * 3 + [_LO] * 7,
        }

    print(f"draft  = {DRAFT_MODEL_NAME}")
    print(f"target = {TARGET_MODEL_NAME}")
    print(f"config = {CONFIG}")
    print(f"seeds  = {SEEDS}, temperature = {TEMPERATURE}, max_new_tokens = {args.max_new_tokens}")
    print(f"traces = {{ {', '.join(f'{k}: {len(v)} rounds' for k, v in TRACES.items())} }}\n", flush=True)

    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)

    print("-- measuring c = T_target / T_draft --", flush=True)
    c_meas = measure_c(draft, target, tok)
    print(f"   c = {c_meas['c']:.2f}  (t_draft={c_meas['t_draft_s']*1e3:.1f}ms, "
          f"t_target={c_meas['t_target_s']*1e3:.1f}ms)\n", flush=True)

    trace_results = []
    for name, trace in TRACES.items():
        print(f"== trace: {name} ({len(trace)} rounds) ==", flush=True)
        tr = run_one_trace(name, trace, draft, target, tok, c_meas["c"], args.max_new_tokens)
        trace_results.append(tr)
        for tax_key, vd in tr["verdicts"].items():
            print(f"   [{tax_key}] {vd['verdict']}", flush=True)
        print(flush=True)

    result = {
        "task": "P5.3",
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "max_new_tokens": args.max_new_tokens,
        "n_prompts": len(PROMPTS),
        "config": vars(CONFIG),
        "batch_signal": ("SYNTHETIC, externally injected -- there is no real high-batch "
                         "throughput on this machine. Real batch curve wiring is M5[A]/cloud."),
        "primary_metric": ("Metric A = cost-model throughput total_emitted/total_cost_units with "
                           "spec round = c+gamma, target step = c (task decision point 6). This "
                           "metric has NO batch-dependent term, so always-spec is an upper bound "
                           "the breaker can only tie -- pitfall 14 resurfacing. Kept as headline "
                           "per sec 9.6 risk 1 (no redefining to win)."),
        "metric_b": ("saturation-aware sensitivity: while batch>=threshold a spec/probe round's "
                     "gamma draft forwards cost sat_tax units each (contention on a saturated "
                     "accelerator, pitfall 7 / Nightjar up-to-30.25% regression). sat_tax in "
                     "{1,2,3}; ~3 reproduces Nightjar's regression order of magnitude for c in "
                     "{4,7,10}. sat_tax=1 == Metric A."),
        "wall_clock_caveat": ("MPS, no KV cache -- wall-clock is unreliable (P1.4: fixed-gamma "
                              "speedup < 1.0 for gamma >= 5). Real throughput is M5[A]/cloud."),
        "measured_c": c_meas,
        "sat_tax_values": SAT_TAX_VALUES,
        "per_trace": trace_results,
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    for tr in trace_results:
        print(f"{tr['trace_name']}: low-batch fraction {tr['low_batch_round_fraction']:.2f}")
        for tax_key, vd in tr["verdicts"].items():
            print(f"  {tax_key}: {vd['verdict']}")
    print(f"\nwritten to {RESULTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
