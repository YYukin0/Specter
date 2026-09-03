"""
P7 Track C -- offline calibration of the linear round-time model
(src/goodput_model.RoundTimeCoeffs) on the real Qwen2.5 0.5B/1.5B pair.

For each grid cell (n_active x prompt_regime x k) we run a few warmup rounds,
then time 5 real `spec_kv_batch.run_round` calls and record one sample per round:

    (n_active, mean_pending, mean_kv_len, k, t_round_s)

`mean_pending` / `mean_kv_len` are measured on entry to the round from the live
SeqStates. A least-squares fit over all samples gives (c0, c1, c2, c3); 20% of
the samples are held out to report R^2 and MAPE.

k = 0 is run as a degraded (plain target) round -- that is the "do not speculate"
arm the controller needs a time estimate for.

Run:
    python src/goodput_profile.py --smoke     # ~1 min, prints coeffs, no file
    python src/goodput_profile.py             # full 72-cell grid, ~5-15 min
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import load_model_and_tokenizer
from prompts import PROMPTS
from rejection_sampling import collect_eos_ids
from spec_kv_batch import make_seq, run_round

import torch

DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
DTYPE = "float16"
SEED_BASE = 1000

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p7_0_goodput_profile.json"
HOME = str(Path.home())

_FILLER_UNIT = ("The following is background context that should be read "
                "carefully before answering. ")
_REGIME_SEGS = {"short": 3, "mid": 18, "long": 36}
_REGIME_TOKENS = {"short": 40, "mid": 250, "long": 500}

N_ACTIVE_GRID = [1, 2, 4, 8]
K_GRID = [0, 1, 2, 3, 5, 8]
REGIMES = ["short", "mid", "long"]

WARMUP_ROUNDS = 3
TIMED_ROUNDS = 5
MAX_NEW_TOKENS = 96


def _prompt_for(regime: str, i: int) -> str:
    base = PROMPTS[i % len(PROMPTS)]
    return _FILLER_UNIT * _REGIME_SEGS[regime] + "\n\nQuestion: " + base


def _mean_pending(seqs) -> float:
    live = [s for s in seqs if not s.done]
    if not live:
        return 0.0
    return sum(len(s.committed) - s.target_synced for s in live) / len(live)


def _mean_kv_len(seqs) -> float:
    live = [s for s in seqs if not s.done]
    if not live:
        return 0.0
    return sum(s.target_synced for s in live) / len(live)


def _run_cell(draft, target, tok, eos_ids, *, n_active: int, regime: str, k: int,
              device=None, make_cache=None, dtype=torch.long):
    """Return a list of (n_active, mean_pending, mean_kv_len, k, t_round_s).

    `device` / `make_cache` are injectable so the hermetic test can drive this
    same code path with make_fake_pair + LengthOnlyCache on CPU.
    """
    device = device if device is not None else torch.device(DEVICE)
    on_mps = device.type == "mps"
    seq_kw = {"make_cache": make_cache} if make_cache is not None else {}
    seqs = [
        make_seq(f"c{i}", _prompt_for(regime, i), tok, device=device,
                 max_new_tokens=MAX_NEW_TOKENS, seed=SEED_BASE + i,
                 apply_chat_template=True, **seq_kw)
        for i in range(n_active)
    ]
    mode = "degraded" if k == 0 else "spec"
    gamma = max(k, 0)

    for _ in range(WARMUP_ROUNDS):
        if all(s.done for s in seqs):
            break
        run_round(seqs, draft, target, gamma=gamma, temperature=1.0,
                  eos_ids=eos_ids, device=device, dtype=dtype, mode=mode)

    samples = []
    for _ in range(TIMED_ROUNDS):
        if all(s.done for s in seqs):
            break
        mp = _mean_pending(seqs)
        kv = _mean_kv_len(seqs)
        if on_mps:
            torch.mps.synchronize()
        t0 = time.perf_counter()
        run_round(seqs, draft, target, gamma=gamma, temperature=1.0,
                  eos_ids=eos_ids, device=device, dtype=dtype, mode=mode)
        if on_mps:
            torch.mps.synchronize()
        t_round = time.perf_counter() - t0
        samples.append((n_active, mp, kv, k, t_round))
    return samples


def _design_matrix(rows, kind: str) -> np.ndarray:
    """rows: list of (n_active, mean_pending, mean_kv_len, k, t). `kind` selects
    which predictor columns to use."""
    cols = []
    for (n, mp, kv, k, _t) in rows:  # noqa: F841
        base = [1.0, n * (mp + k), n * kv, n * k]
        if kind == "plus_nactive":
            base.append(float(n))
        cols.append(base)
    return np.asarray(cols, dtype=float)


def _nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares (active-set, no reactivation). Every column of
    the round-time model is a physical cost -> its coefficient cannot be
    negative. numpy has no NNLS and scipy is not a dependency, so this small
    projected solver stands in: solve unconstrained, pin the most-negative
    coefficient to 0, refit the rest, repeat.
    """
    ncol = X.shape[1]
    zero: set[int] = set()
    beta = np.zeros(ncol)
    for _ in range(ncol + 1):
        free = [c for c in range(ncol) if c not in zero]
        if not free:
            return np.zeros(ncol)
        b, *_ = np.linalg.lstsq(X[:, free], y, rcond=None)
        beta = np.zeros(ncol)
        for i, c in enumerate(free):
            beta[c] = b[i]
        neg = [c for c in free if beta[c] < -1e-12]
        if not neg:
            return beta
        zero.add(min(neg, key=lambda c: beta[c]))
    return beta


def _fit(rows, kind: str, rng: random.Random):
    X = _design_matrix(rows, kind)
    y = np.asarray([r[4] for r in rows], dtype=float)
    n = len(rows)
    idx = list(range(n))
    rng.shuffle(idx)
    n_hold = max(1, n // 5)
    hold = set(idx[:n_hold])
    tr = [i for i in idx if i not in hold]
    ho = [i for i in idx if i in hold]

    beta = _nnls(X[tr], y[tr])
    pred_ho = X[ho] @ beta
    y_ho = y[ho]
    ss_res = float(np.sum((y_ho - pred_ho) ** 2))
    ss_tot = float(np.sum((y_ho - np.mean(y_ho)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs((y_ho - pred_ho) / np.clip(np.abs(y_ho), 1e-9, None))))
    return beta, r2, mape, len(tr), len(ho)


def _fit_and_report(rows):
    """rows: list of (n_active, mean_pending, mean_kv_len, k, t_round_s).
    Returns the full result dict (minus elapsed_s_total)."""
    rng = random.Random(SEED_BASE)
    beta, r2, mape, n_tr, n_ho = _fit(rows, "base", rng)
    design = "T = c0 + c1*(n_active*(mean_pending+k)) + c2*(n_active*mean_kv_len) + c3*(n_active*k)"
    kind = "base"
    if r2 < 0.85:
        rng2 = random.Random(SEED_BASE + 1)
        beta2, r2_2, mape2, n_tr2, n_ho2 = _fit(rows, "plus_nactive", rng2)
        if r2_2 > r2:
            beta, r2, mape, n_tr, n_ho = beta2, r2_2, mape2, n_tr2, n_ho2
            design += " + c4*n_active   (NNLS, added: held-out R^2 improved)"
            kind = "plus_nactive"

    coeffs = {
        "c0": float(beta[0]), "c1": float(beta[1]),
        "c2": float(beta[2]), "c3": float(beta[3]),
        "r2": float(r2), "n_fit": int(n_tr),
    }
    if kind == "plus_nactive":
        coeffs["c4_n_active"] = float(beta[4])

    # collinearity diagnostic: mean_pending is pinned at 1 by the rectangular
    # invariant, so n*(mean_pending+k) and n*k differ only by n -> c1 and c3 are
    # not separately identifiable and the NNLS fit pins c3 at its 0 boundary.
    mp_vals = sorted({round(r[1], 2) for r in rows})
    pending_constant = mp_vals == [1.0]
    notes = []
    if r2 < 0.85:
        notes.append(f"held-out R^2 {r2:.3f} < 0.85: a 4-param linear model is a "
                     f"coarse fit to {len(rows)} MPS round timings.")
    if coeffs["c3"] <= 1e-9:
        notes.append(
            "c3 (draft-only per-token cost) pinned at 0 by NNLS. Cause: the "
            "rectangular-invariant serving loop keeps mean_pending == 1 every "
            "round (mp_vals=" + str(mp_vals) + "), so n*(mean_pending+k) and "
            "n*k are collinear and the verify/draft split is unidentifiable. "
            f"The combined per-speculative-token cost lives in c1={coeffs['c1']:.4f} "
            "s/token. The draft cost is real (Pitfall 14) but on this 0.5B/1.5B "
            "pair it is a small fixed fraction folded into c1 -- consistent with "
            "the Mac dead-zone parity finding. A pair with an expensive draft, or "
            "a profiler that varies pending depth, would separate them. best_k "
            "only needs relative goodput, which c1 alone captures here.")
    if coeffs["c1"] <= 1e-9:
        notes.append(f"WARNING: c1={coeffs['c1']:.3e} <= 0 -- round time not "
                     f"increasing in speculative work; controller output suspect.")

    return {
        "task": "P7.0 goodput linear round-time calibration",
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "device": DEVICE, "dtype": DTYPE,
        "grid": {"n_active": N_ACTIVE_GRID, "regime_tokens": _REGIME_TOKENS, "k": K_GRID},
        "n_samples": len(rows), "n_heldout": n_ho,
        "design": design,
        "fit_method": "non-negative least squares (active-set), 20% held out",
        "mean_pending_constant": pending_constant,
        "coeffs": coeffs,
        "heldout_mape": float(mape),
        "raw_samples": [
            {"n_active": r[0], "mean_pending": round(r[1], 3),
             "mean_kv_len": round(r[2], 3), "k": r[3], "t_round_s": round(r[4], 6)}
            for r in rows
        ],
        "acceptance_note": " ".join(notes),
    }


def run(smoke: bool):
    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)
    eos_ids = collect_eos_ids(tok, target)

    if smoke:
        n_grid, regimes, k_grid = [1, 4], ["short"], [0, 3]
        global WARMUP_ROUNDS, TIMED_ROUNDS
        WARMUP_ROUNDS, TIMED_ROUNDS = 1, 2
    else:
        n_grid, regimes, k_grid = N_ACTIVE_GRID, REGIMES, K_GRID

    rows = []
    for regime in regimes:
        for n_active in n_grid:
            for k in k_grid:
                cell = _run_cell(draft, target, tok, eos_ids,
                                 n_active=n_active, regime=regime, k=k)
                rows.extend(cell)
                if cell:
                    mt = sum(c[4] for c in cell) / len(cell)
                    print(f"[{regime:5s}] n={n_active} k={k}: {len(cell)} rounds, "
                          f"mean t_round={mt*1e3:.1f}ms")

    return _fit_and_report(rows)


def refit():
    """Recompute coeffs + notes from an existing results file's raw_samples,
    without re-timing anything."""
    blob = json.loads(RESULTS_PATH.read_text())
    rows = [(s["n_active"], s["mean_pending"], s["mean_kv_len"], s["k"], s["t_round_s"])
            for s in blob["raw_samples"]]
    return _fit_and_report(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--refit", action="store_true",
                    help="recompute coeffs from the existing results file's raw_samples")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.refit:
        out = refit()
    else:
        out = run(args.smoke)
    out["elapsed_s_total"] = time.perf_counter() - t0

    print(json.dumps(out["coeffs"], indent=2))
    print(f"heldout_mape={out['heldout_mape']:.3f}  n_samples={out['n_samples']}")
    if out["acceptance_note"]:
        print("acceptance_note:", out["acceptance_note"])

    if not args.smoke:
        text = json.dumps(out, indent=2).replace(HOME, "~")
        RESULTS_PATH.write_text(text)
        print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
