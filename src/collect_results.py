"""
Roll every results/*.json up into one markdown table for quick review and
report drafting.

Not a re-analysis: this only *reads* the result files each experiment
already wrote and pulls out (P-number, headline metric, number, one-line
takeaway, file path). Per-file extractors below know the schema of the
main P-numbered outputs; anything without a registered extractor falls
through to a generic best-effort reader (looks for task / verdict /
conclusion / *_pass / gate-ish keys) and is clearly marked "(generic)".

Run:  python src/collect_results.py [--out notes/results-summary_<date>.md]
Default out: notes/results-summary_2026-08-28.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
DEFAULT_OUT = REPO / "notes" / "results-summary_2026-08-28.md"


def _fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _row(pnum, metric, number, takeaway):
    return {"p": pnum, "metric": metric, "number": number, "takeaway": takeaway}


# --------------------------------------------------------------------------- #
# per-file extractors: stem -> fn(dict) -> row dict
# --------------------------------------------------------------------------- #
def _x_p1_0_b(d):
    return _row("P1.0 [B]", "overall_alpha", _fmt(d.get("overall_alpha")),
               f"{d.get('gate_tier')} -- {d.get('conclusion')}")


def _x_p1_0_a(d):
    g = d.get("gate", {})
    return _row("P1.0 [A]", "greedy_agreement / accept",
               f"{d.get('greedy_agreement')} / {d.get('pytorch_acceptance')}",
               f"gate={g if not isinstance(g, dict) else g.get('status', g)}")


def _x_p1_0_mlx(d):
    return _row("P1.0 mlx-crosscheck", "overall_alpha", _fmt(d.get("overall_alpha")),
               d.get("note", "")[:160])


def _x_p1_2(d):
    return _row("P1.2", "p1_2_pass", str(d.get("p1_2_pass")),
               "greedy verifier: forward+reverse fault-injection both hold"
               if d.get("p1_2_pass") else "greedy verifier FAILED")


def _x_p1_3(d):
    sa = d.get("statistical_alpha", {})
    return _row("P1.3", "empirical / theoretical alpha",
               f"{_fmt(sa.get('empirical_alpha'))} / {_fmt(sa.get('theoretical_alpha'))}"
               if isinstance(sa, dict) else _fmt(sa),
               f"agrees_within_2se={sa.get('agrees_within_2se') if isinstance(sa, dict) else '?'}; "
               f"pass={d.get('p1_3_pass')}")


def _x_p1_4(d):
    pg = d.get("per_gamma", [])
    best = None
    for v in pg if isinstance(pg, list) else []:
        mean = v.get("emitted_per_round_mean")
        if mean is not None and (best is None or mean > best[1]):
            best = (v.get("gamma"), mean, v.get("speedup_vs_target_only_mean"))
    return _row("P1.4", "gamma sweep -> best emitted/round",
               f"gamma={best[0]}: {_fmt(best[1])} (speedup {_fmt(best[2], 2)}x)" if best else "see file",
               "emitted/round monotone in gamma; accept-length std also grows with gamma")


def _x_p2_0(d):
    sce = d.get("salient_channel_evidence", {})
    top = (sce.get("top5_layers") or [{}])[0] if isinstance(sce, dict) else {}
    return _row("P2.0", "n target linears / top salient ratio",
               f"{d.get('n_target_layers_collected')} layers; "
               f"max abs_mean max/median = {_fmt(top.get('ratio'), 1)} ({top.get('layer', '?')})",
               "activation stats collected; early mlp.down_proj extremely peaked -> AWQ has something to protect")


def _x_p2_1(d):
    s = d.get("summary", {})
    return _row("P2.1", "layers quantized / fell back",
               f"{s.get('n_quantized')} quant / {s.get('n_fell_back')} fp16 / {s.get('n_skipped')} skip",
               f"whole-model fake-quant pipeline; timing {d.get('timing_s')}s")


def _x_p2_2(d):
    de = d.get("deltas", {})
    v = d.get("verdict")
    v0 = v[0] if isinstance(v, list) and v else v
    return _row("P2.2", "AWQ ppl delta vs fp16, same vs cross calib (eval=NL)",
               f"same +{_fmt(de.get('same_dist_delta_NL'))} vs cross +{_fmt(de.get('cross_dist_delta_NL'))} "
               f"(cross-minus-same +{_fmt(de.get('cross_minus_same_NL'))})"
               if isinstance(de, dict) else "see file",
               str(v0)[:200])


def _x_p2_3(d):
    knee = d.get("knee_small_minus_large") or d.get("knee")
    return _row("P2.3", "calib-size vs wikitext2 ppl",
               f"knee(small-large)={_fmt(knee)}" if knee is not None else "see file",
               str(d.get("verdict") or d.get("conclusion") or "")[:200])


def _mean_gen_tps(block):
    tps = [r.get("generation_tps") for p in (block.get("per_prompt") or [])
           for r in (p.get("runs") or []) if r.get("generation_tps")]
    return sum(tps) / len(tps) if tps else None


def _x_p2_4_mlx(d):
    b = _mean_gen_tps(d.get("baseline_bf16", {}) or {})
    q = _mean_gen_tps(d.get("awq_int4_g128", {}) or {})
    spd = f"{q / b:.2f}x" if b and q else "?"
    return _row("P2.4", "mlx_lm.awq int4 vs bf16 gen tok/s (3B)",
               f"bf16 {_fmt(b, 1)} -> int4 {_fmt(q, 1)} ({spd})",
               (d.get("note", "") or "")[:180])


def _x_p2_awq_michael(d):
    return _row("P2.2/2.3 [Michael 3B, ref only]", "baseline ppl",
               _fmt(d.get("baseline_perplexity")),
               f"target={d.get('target_model')} dtype={d.get('dtype')} -- direction-only reference, NOT comparable")


def _x_p4_0(d):
    curve = d.get("curve") or d.get("points") or []
    return _row("P4.0 [B]", "quant throughput curve pts",
               f"{len(curve)} points" if hasattr(curve, "__len__") else "see file",
               d.get("scope_note", "")[:160])


def _x_p5_0(d):
    v = d.get("verdict", {})
    return _row("P5.0", "GammaTune vs best fixed (mean_emitted/round)",
               (f"{_fmt((v.get('gammatune_mean_emitted_per_round') or [None])[0])} vs "
                f"{_fmt((v.get('best_fixed_mean_emitted_per_round') or [None])[0])}")
               if isinstance(v, dict) else "see file",
               str(v.get("verdict") if isinstance(v, dict) else v)[:200])


def _x_p5_1(d):
    ss = d.get("steady_state_comparison", {})
    return _row("P5.1", "non-stationary steady-state",
               str(ss)[:80] if ss else "see file",
               "GammaTune tracked across A/B segment switches; see file for per-sequence")


def _x_p5_3(d):
    return _row("P5.3", "circuit breaker Metric A / sat_tax",
               f"sat_tax values {d.get('sat_tax_values')}",
               "always-spec is structural upper bound (pitfall 15); breaker null")


def _x_p5_4(d):
    return _row("P5.4", "HF assisted-gen baseline",
               str(d.get("primary_metric"))[:80],
               str(d.get("conclusion", ""))[:200])


def _x_p5_5(d):
    obs = d.get("observations", {})
    cur = d.get("curves", {})
    return _row("P5.5 [A]", "tokens/target-forward vs batch {1,2,4,8}",
               str(cur.get("tokens_per_target_forward_vs_batch", "see file"))[:120],
               f"rises_with_batch={obs.get('tokens_per_target_forward_rises_with_batch')}, "
               f"eff_drop={_fmt(obs.get('efficiency_drop_bs1_to_bsN'))}")


def _x_p3_1(d):
    cmp = d.get("comparison", {})
    return _row("P3.1", "structured vs freetext alpha",
               f"{_fmt((d.get('structured') or {}).get('pooled_alpha'))} vs "
               f"{_fmt((d.get('freetext') or {}).get('pooled_alpha'))}",
               str(cmp.get("verdict", ""))[:200])


def _x_worse_pair_alpha(d):
    return _row(f"side-exp {d.get('pair_tag', '?')} alpha", "per-gamma alpha sweep",
               str(d.get("reading", ""))[:100],
               str(d.get("_note", d.get("reading", "")))[:200])


def _x_worse_pair_gt(d):
    v = d.get("verdict", {})
    return _row(f"side-exp {d.get('pair_tag', '?')} GammaTune", "verdict",
               str(v.get("verdict") if isinstance(v, dict) else v)[:100],
               str(d.get("conclusion", ""))[:200])


EXTRACTORS = {
    "p1_0_gate_result_b_track": _x_p1_0_b,
    "p1_0_gate_result": _x_p1_0_a,
    "p1_0_mlx_crosscheck_result": _x_p1_0_mlx,
    "p1_2_greedy_verifier": _x_p1_2,
    "p1_3_sampling_verifier": _x_p1_3,
    "p1_4_gamma_sweep": _x_p1_4,
    "p2_0_activation_stats": _x_p2_0,
    "p2_1_full_quant": _x_p2_1,
    "p2_2_cross_distribution": _x_p2_2,
    "p2_3_calib_size": _x_p2_3,
    "p2_4_mlx_awq_crosscheck": _x_p2_4_mlx,
    "p2_awq_calibration_michael_3b": _x_p2_awq_michael,
    "p3_1_structured_vs_freetext_alpha": _x_p3_1,
    "p4_0_quant_throughput_b_track": _x_p4_0,
    "p5_0_gammatune": _x_p5_0,
    "p5_1_nonstationary": _x_p5_1,
    "p5_3_circuit_breaker": _x_p5_3,
    "p5_4_hf_baseline": _x_p5_4,
    "p5_5_spec_batch_curve": _x_p5_5,
    "explore_worse_pair_pair1_alpha": _x_worse_pair_alpha,
    "explore_worse_pair_pair2_alpha": _x_worse_pair_alpha,
    "explore_worse_pair_pair1_gammatune": _x_worse_pair_gt,
    "explore_worse_pair_pair2_gammatune": _x_worse_pair_gt,
}

# scaling_demo is the superseded first-pass of P2.1 -- list it, flag it
_SUPERSEDED = {"p2_1_scaling_demo": "superseded by p2_1_full_quant"}


def _generic(d):
    if not isinstance(d, dict):
        return _row("?", "(generic) non-dict json", type(d).__name__, "")
    metric = d.get("task") or d.get("experiment") or "(generic)"
    number = ""
    for k in ("overall_alpha", "alpha", "perplexity", "p1_2_pass", "p1_3_pass"):
        if k in d:
            number = f"{k}={_fmt(d[k])}"
            break
    takeaway = ""
    for k in ("conclusion", "verdict", "reading", "note", "_note", "gate_tier"):
        if k in d:
            takeaway = str(d[k])[:200]
            break
    return _row(str(metric)[:60], "(generic)", number, takeaway)


def collect():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        stem = path.stem
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            rows.append((stem, _row("?", "UNREADABLE", str(e)[:80], ""), path))
            continue
        fn = EXTRACTORS.get(stem)
        row = fn(d) if fn else _generic(d)
        if stem in _SUPERSEDED:
            row["takeaway"] = f"[{_SUPERSEDED[stem]}] " + row["takeaway"]
        if fn is None and stem not in _SUPERSEDED:
            row["metric"] = row["metric"] + " (generic)"
        rows.append((stem, row, path))
    return rows


def render(rows) -> str:
    lines = [
        "# Specter results summary",
        "",
        f"Auto-generated by `src/collect_results.py` from `results/*.json` "
        f"({len(rows)} files). Regenerate after any new experiment; do not hand-edit.",
        "",
        "| P / experiment | metric | number | one-line takeaway | file |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _stem, r, path in rows:
        rel = path.relative_to(REPO)
        cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {cell(r['p'])} | {cell(r['metric'])} | {cell(r['number'])} "
            f"| {cell(r['takeaway'])} | `{rel}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = collect()
    text = render(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(f"\nwrote {args.out.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
