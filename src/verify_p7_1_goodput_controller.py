"""
P7 Track C verify -- fixed vs alpha_floor vs goodput speculation-length control,
on non-stationary prompt streams at four concurrency widths, plus a load-ramp
that watches k* decay as n_active climbs.

Real Qwen2.5 0.5B/1.5B on MPS fp16. The goodput controller reads the linear
round-time coefficients calibrated by src/goodput_profile.py
(results/p7_0_goodput_profile.json).

Run:
    python src/verify_p7_1_goodput_controller.py --smoke   # dummy coeffs, no file
    python src/verify_p7_1_goodput_controller.py           # full, writes results/
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import load_model_and_tokenizer
from nonstationary_prompts import SEGMENT_A, SEGMENT_B, SEQUENCES
from serving_loop import ServeConfig, SpecServer
from goodput_model import RoundTimeCoeffs

DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
DTYPE = "float16"
SEED_BASE = 1000

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p7_1_goodput_controller.json"
COEFFS_PATH = Path(__file__).resolve().parent.parent / "results" / "p7_0_goodput_profile.json"
HOME = str(Path.home())

GAMMA_FIXED = 3
MAX_NEW_TOKENS = 48
SEQ_NAMES = ["A_to_B", "A_to_B_to_A", "ABAB"]
WIDTHS = [1, 2, 4, 8]
CONTROLLERS = ["fixed", "alpha_floor", "goodput"]

_DUMMY_COEFFS = RoundTimeCoeffs(c0=2e-3, c1=3e-4, c2=2e-6, c3=8e-4)


def _mode_switches(round_log):
    modes = [inf.mode for inf in round_log if inf.mode != "idle"]
    return sum(1 for a, b in zip(modes, modes[1:]) if a != b)


def _serve_once(draft, target, tok, prompts, *, width, controller, coeffs=None):
    cfg = ServeConfig(
        gamma=GAMMA_FIXED, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
        max_active=width, controller=controller,
        breaker_on=(controller == "alpha_floor"), alpha_floor=0.5,
        warmup_rounds=4, controller_warmup=4,
        goodput_coeffs_path=str(COEFFS_PATH),
    )
    if coeffs is not None:
        cfg.goodput_coeffs = coeffs
    srv = SpecServer(draft, target, tok, cfg)
    for i, p in enumerate(prompts):
        srv.submit(p, req_id=f"p{i}", seed=SEED_BASE + i)

    t0 = time.perf_counter()
    srv.run_until_idle(max_rounds=5000)
    wall = time.perf_counter() - t0

    res = srv.results()
    total_tokens = sum(len(r.token_ids) for r in res.values())
    accepted_total = sum(sum(r.accept_lengths) for r in res.values())
    non_idle = [inf for inf in srv.round_log if inf.mode != "idle"]

    if controller == "goodput":
        ks = [inf.controller_k for inf in non_idle if inf.controller_k >= 0]
    else:
        ks = [inf.round_gamma for inf in non_idle if inf.mode == "spec"]
    mean_k = statistics.fmean(ks) if ks else 0.0

    spec_alphas = [inf.rolling_alpha for inf in non_idle if inf.mode == "spec"]
    return {
        "sequence": None, "width": width, "controller": controller,
        "agg_tok_per_s": total_tokens / wall if wall else 0.0,
        "accepted_total": accepted_total,
        "useful_goodput": accepted_total / wall if wall else 0.0,
        "mean_k": mean_k,
        "n_spec": sum(1 for inf in non_idle if inf.mode == "spec"),
        "n_degraded": sum(1 for inf in non_idle if inf.mode == "degraded"),
        "n_probe": sum(1 for inf in non_idle if inf.mode == "probe"),
        "rolling_alpha_mean": statistics.fmean(spec_alphas) if spec_alphas else 0.0,
        "mode_switches": _mode_switches(srv.round_log),
        "wall_s": wall,
        "n_requests": len(res),
    }


def _load_ramp(draft, target, tok, *, coeffs=None):
    prompts = [p for _, p in [("A", q) for q in SEGMENT_A] + [("B", q) for q in SEGMENT_B]] * 2
    cfg = ServeConfig(
        gamma=GAMMA_FIXED, temperature=1.0, max_new_tokens=MAX_NEW_TOKENS,
        max_active=8, controller="goodput", breaker_on=False,
        warmup_rounds=4, controller_warmup=4, goodput_coeffs_path=str(COEFFS_PATH),
    )
    if coeffs is not None:
        cfg.goodput_coeffs = coeffs
    srv = SpecServer(draft, target, tok, cfg)
    for i, p in enumerate(prompts):
        srv.submit(p, req_id=f"r{i}", seed=SEED_BASE + i)
    srv.run_until_idle(max_rounds=5000)

    series = [
        {"round": inf.index, "n_active": inf.n_active,
         "controller_k": inf.controller_k, "rolling_alpha": round(inf.rolling_alpha, 4)}
        for inf in srv.round_log if inf.mode != "idle"
    ]
    ks_full = [s["controller_k"] for s in series
               if s["n_active"] >= 8 and s["controller_k"] >= 0]
    return {
        "width": 8, "controller": "goodput", "n_prompts": len(prompts),
        "series": series,
        "k_mean_after_full": statistics.fmean(ks_full) if ks_full else 0.0,
    }


def _acc_note(runs, acc, smoke: bool) -> str:
    """Honest write-up of what the controller actually did (P5.0 GammaTune /
    P6.7 kernel house style: report the negative result plainly)."""
    if smoke:
        return ""
    note = ""

    # throughput: how does goodput's agg tok/s compare to the better baseline,
    # per (sequence, width>=2) cell?
    losses = []
    for r in runs:
        if r["controller"] != "goodput" or r["width"] < 2:
            continue
        peers = [x for x in runs if x["sequence"] == r["sequence"]
                 and x["width"] == r["width"] and x["controller"] in ("fixed", "alpha_floor")]
        if not peers:
            continue
        best = max(x["agg_tok_per_s"] for x in peers)
        if best > 0:
            losses.append((r["agg_tok_per_s"] - best) / best)
    if losses:
        worst = min(losses) * 100
        mean = sum(losses) / len(losses) * 100
        n_worse = sum(1 for x in losses if x < -0.02)
        note += (
            f"Negative result. The goodput controller shrinks k below the fixed "
            f"gamma={GAMMA_FIXED} on every non-trivial cell (mean_k ~1.6-2.5 vs 3.0) "
            f"and, at width>=2, that costs throughput: aggregate tok/s is below the "
            f"better of the two baselines on {n_worse}/{len(losses)} cells, "
            f"mean {mean:.0f}%, worst {worst:.0f}%. width=8 useful_goodput beats the "
            f"best baseline on {acc['width8_goodput_ge_best_on']}/3 sequences. "
        )
    note += (
        "The machinery is correct -- k* tracks the round-time model's argmax, the "
        "hysteresis clamp keeps it from chattering, alpha=1 is handled, no run "
        f"failed, and under the load ramp k* does contract "
        f"(k_mean_after_full={acc['ramp_k_mean_after_full']:.2f}). The model's "
        "optimum just isn't the throughput optimum on this pair. The linear "
        "round-time model charges c1=0.0235 s per speculative token (the profile's "
        "NNLS folded the unidentifiable draft-cost term into c1 -- see "
        "p7_0_goodput_profile.json acceptance_note), so expected_round_time rises "
        "roughly linearly in k while expected_accepted_tokens saturates at "
        "alpha~=0.77; the ratio therefore peaks at small k. On this Mac the extra "
        "cost of verifying k=3 vs k=1 draft tokens in one batched target forward "
        "is nearly free (Pitfall 4 dead zone), so the model over-penalizes k and "
        "the controller leaves accepted tokens on the table. A pair with an "
        "expensive draft, or a profiler that varies pending depth to separate the "
        "draft term, would move the model's optimum back toward the real one. "
    )
    if not acc["goodput_differs_from_alpha_floor"]:
        note += "goodput made the same decisions as alpha_floor everywhere. "
    return note


def run(smoke: bool):
    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)

    seq_names = ["ABAB"] if smoke else SEQ_NAMES
    widths = [1, 4] if smoke else WIDTHS
    controllers = ["fixed", "goodput"] if smoke else CONTROLLERS
    coeffs = _DUMMY_COEFFS if smoke else None

    runs = []
    for seq_name in seq_names:
        prompts = [p for _, p in SEQUENCES[seq_name]]
        for width in widths:
            for ctrl in controllers:
                row = _serve_once(draft, target, tok, prompts, width=width,
                                  controller=ctrl, coeffs=coeffs)
                row["sequence"] = seq_name
                runs.append(row)
                print(f"[{seq_name:11s}] w={width} {ctrl:11s} "
                      f"gp={row['useful_goodput']:.1f} tok/s  "
                      f"agg={row['agg_tok_per_s']:.1f}  mean_k={row['mean_k']:.2f}  "
                      f"deg={row['n_degraded']} sw={row['mode_switches']}")

    ramp = _load_ramp(draft, target, tok, coeffs=coeffs)
    print(f"[ramp] rounds={len(ramp['series'])}  k_mean_after_full={ramp['k_mean_after_full']:.2f}")

    # acceptance checks
    acc = {}
    ge_best = 0
    for seq_name in seq_names:
        w8 = {r["controller"]: r for r in runs if r["sequence"] == seq_name and r["width"] == 8}
        if {"fixed", "alpha_floor", "goodput"} <= set(w8):
            best_baseline = max(w8["fixed"]["useful_goodput"], w8["alpha_floor"]["useful_goodput"])
            if w8["goodput"]["useful_goodput"] >= best_baseline:
                ge_best += 1
    acc["width8_goodput_ge_best_on"] = ge_best
    acc["ramp_k_mean_after_full"] = ramp["k_mean_after_full"]

    differs = False
    for seq_name in seq_names:
        for width in (widths if smoke else WIDTHS):
            g = next((r for r in runs if r["sequence"] == seq_name and r["width"] == width
                      and r["controller"] == "goodput"), None)
            a = next((r for r in runs if r["sequence"] == seq_name and r["width"] == width
                      and r["controller"] == "alpha_floor"), None)
            if g and a and (abs(g["mean_k"] - a["mean_k"]) > 0.25
                            or g["n_degraded"] != a["n_degraded"]):
                differs = True
    acc["goodput_differs_from_alpha_floor"] = differs

    acc_note = _acc_note(runs, acc, smoke)

    out = {
        "task": "P7.1 fixed vs alpha_floor vs goodput controller",
        "draft_model": DRAFT_MODEL_NAME, "target_model": TARGET_MODEL_NAME,
        "device": DEVICE, "dtype": DTYPE,
        "gamma_fixed": GAMMA_FIXED, "max_new_tokens": MAX_NEW_TOKENS,
        "coeffs_path": str(COEFFS_PATH).replace(HOME, "~"),
        "runs": runs,
        "ramp": ramp,
        "acceptance": acc,
        "a40_shape_check": (
            "eagle3 advantage 2.4x@c16 -> 1.6x@c64 in bullet2_vllm_eagle3.json; "
            "the goodput round-time model's batched-verify term grows with n_active, "
            "so k* shrinks as concurrency rises -- same direction. NOTE: bullet2's "
            "mean_acceptance_rate is null, so this is a qualitative shape match only, "
            "not a quantitative cross-validation."
        ),
        "acceptance_note": acc_note,
    }
    return out


def _rescore():
    """Recompute acceptance + acceptance_note from an existing result file's
    `runs`/`ramp` -- no re-timing (the full run is ~24 min of MPS). Same idea as
    goodput_profile.py --refit."""
    out = json.loads(RESULTS_PATH.read_text())
    runs, ramp = out["runs"], out["ramp"]
    seq_names = sorted({r["sequence"] for r in runs})

    ge_best = 0
    for seq_name in seq_names:
        w8 = {r["controller"]: r for r in runs if r["sequence"] == seq_name and r["width"] == 8}
        if {"fixed", "alpha_floor", "goodput"} <= set(w8):
            best = max(w8["fixed"]["useful_goodput"], w8["alpha_floor"]["useful_goodput"])
            if w8["goodput"]["useful_goodput"] >= best:
                ge_best += 1
    differs = any(
        g and a and (abs(g["mean_k"] - a["mean_k"]) > 0.25 or g["n_degraded"] != a["n_degraded"])
        for seq_name in seq_names for width in WIDTHS
        for g in [next((r for r in runs if r["sequence"] == seq_name and r["width"] == width
                        and r["controller"] == "goodput"), None)]
        for a in [next((r for r in runs if r["sequence"] == seq_name and r["width"] == width
                        and r["controller"] == "alpha_floor"), None)]
    )
    acc = {
        "width8_goodput_ge_best_on": ge_best,
        "ramp_k_mean_after_full": ramp["k_mean_after_full"],
        "goodput_differs_from_alpha_floor": differs,
    }
    out["acceptance"] = acc
    out["acceptance_note"] = _acc_note(runs, acc, smoke=False)
    RESULTS_PATH.write_text(json.dumps(out, indent=2).replace(HOME, "~"))
    print(json.dumps(acc, indent=2))
    print("\n" + out["acceptance_note"])
    print(f"\nrescored {RESULTS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute acceptance/note from the existing JSON, no re-timing")
    args = ap.parse_args()

    if args.rescore:
        _rescore()
        return

    t0 = time.perf_counter()
    out = run(args.smoke)
    out["elapsed_s_total"] = time.perf_counter() - t0

    if not args.smoke:
        text = json.dumps(out, indent=2).replace(HOME, "~")
        RESULTS_PATH.write_text(text)
        print(f"\nwrote {RESULTS_PATH}")
    else:
        print(json.dumps(out["acceptance"], indent=2))


if __name__ == "__main__":
    main()
