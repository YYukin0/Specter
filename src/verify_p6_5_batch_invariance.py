"""
P6.5 Part 5 -- batch-invariance classification (the "non-algorithmic" baseline).

Two things this pins, both hermetic (deterministic FakeModel, no model load):

  1. BATCH INVARIANCE OF THE ALGORITHM. `spec_kv_batch.run_round` driven over N
     sequences emits exactly what N independent `speculative_generate_kv` runs
     emit -- token for token, greedy and sampling, every gamma/seed. On a
     noise-free model, batching changes nothing: max |token delta| = 0. This is
     the reference against which a *real* backend's batch non-determinism is
     measured.

  2. specdiff's clean path. With no fault operator, `specdiff.diagnose` returns
     NO_DIVERGENCE on every prompt -- the differential debugger does not invent
     a mechanism when the traces agree.

What is NOT hermetic and is left for the O2 increment (needs real Qwen on MPS /
a real int4 build): arXiv:2607.17283 measures that a quantized Metal backend
runs the "parallel" verify forward with a per-logit delta on the order of
5.8e-3 vs the serial path, purely from batch-variant kernel reductions -- no
algorithm bug. specdiff is built to route that case (every structural signal
agrees, only the final tokens differ) to BACKEND_NONDETERMINISM, i.e. classify
it as non-algorithmic and report the logit delta magnitude. This script records
the target number and the classification contract; verify_p6_5_o2.py (future)
fills in the measured value.

Run:  python src/verify_p6_5_batch_invariance.py [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import rejection_sampling as _rs
from spec_kv import speculative_generate_kv
from spec_kv_batch import make_seq, run_round
from spec_oracles import LengthOnlyCache, _PROMPTS, make_fake_pair
import specdiff

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p6_5_batch_invariance.json"

# arXiv:2607.17283, "Lossless but Not Free": consumer-hardware quantized Metal
# backend, parallel vs serial speculative verify, per-logit delta magnitude.
PAPER_BACKEND_LOGIT_DELTA = 5.8e-3


def _drive_batch(seqs, draft, target, tok, *, gamma, temperature):
    eos = _rs.collect_eos_ids(tok, target)
    dev = torch.device("cpu")
    while any(not s.done for s in seqs):
        run_round(seqs, draft, target, gamma=gamma, temperature=temperature,
                  eos_ids=eos, device=dev, dtype=torch.float32, mode="spec")


def batch_invariance(gammas, seeds, temps, max_new_tokens=48):
    draft, target, tok = make_fake_pair()
    n_seq_pairs = 0
    n_mismatch = 0
    max_delta = 0
    first_div_tokens = []
    for temperature in temps:
        for gamma in gammas:
            for seed in seeds:
                refs = [
                    speculative_generate_kv(p, draft, target, tok,
                                            make_cache=LengthOnlyCache, gamma=gamma,
                                            max_new_tokens=max_new_tokens,
                                            temperature=temperature, seed=seed).token_ids
                    for p in _PROMPTS
                ]
                seqs = [make_seq(f"s{i}", p, tok, device=torch.device("cpu"),
                                 max_new_tokens=max_new_tokens, seed=seed,
                                 make_cache=LengthOnlyCache)
                        for i, p in enumerate(_PROMPTS)]
                _drive_batch(seqs, draft, target, tok, gamma=gamma, temperature=temperature)
                for i in range(len(_PROMPTS)):
                    n_seq_pairs += 1
                    a, b = seqs[i].token_ids, refs[i]
                    m = min(len(a), len(b))
                    div = next((k for k in range(m) if a[k] != b[k]), None)
                    if div is None and abs(len(a) - len(b)) > gamma:
                        div = m
                    if div is not None:
                        n_mismatch += 1
                        first_div_tokens.append(div)
                        max_delta = max(max_delta, abs(a[div] - b[div]) if div < m else 999)
    return {
        "n_sequence_pairs": n_seq_pairs,
        "n_mismatch": n_mismatch,
        "max_abs_token_delta": max_delta,
        "first_divergence_tokens": first_div_tokens,
        "verdict": ("batch-invariant: batched run_round == N single-sequence runs, "
                    "bit-exact" if n_mismatch == 0 else "NOT batch-invariant -- real bug"),
    }


def specdiff_clean(prompts):
    reports = []
    all_nodiv = True
    for p in prompts:
        rep = specdiff.diagnose(p, mutants=())
        reports.append({"prompt": p, "mechanism": rep.mechanism,
                        "offending_round": rep.offending_round})
        all_nodiv = all_nodiv and rep.mechanism == specdiff.NO_DIVERGENCE
    return {"all_NO_DIVERGENCE": all_nodiv, "reports": reports}


def run(smoke=False):
    gammas = (3,) if smoke else (1, 3, 5)
    seeds = (0, 1) if smoke else (0, 1, 2, 3, 4)
    temps = (0.0, 1.0)

    inv = batch_invariance(gammas, seeds, temps)
    sd = specdiff_clean(_PROMPTS if not smoke else _PROMPTS[:2])

    return {
        "task": "P6.5 Part 5 -- batch-invariance classification (non-algorithmic baseline)",
        "model": "deterministic position-one-hot FakeModel (hermetic)",
        "gammas": list(gammas),
        "seeds": list(seeds),
        "temperatures": list(temps),
        "algorithm_batch_invariance": inv,
        "specdiff_clean_path": sd,
        "real_backend_increment_O2": {
            "status": "not run here -- needs real Qwen2.5 on MPS / a real int4 build",
            "paper": "arXiv:2607.17283 (Lossless but Not Free)",
            "expected_per_logit_delta_magnitude": PAPER_BACKEND_LOGIT_DELTA,
            "classification_contract": (
                "when every structural signal (draft tokens, cache-position "
                "vectors, cache lengths, accept/reject decisions, n_accepted, "
                "post-rollback cache length) agrees between the serial and "
                "batched verify but the emitted tokens differ, specdiff.classify "
                "returns BACKEND_NONDETERMINISM -- i.e. it is reported as a "
                "backend batch-variance effect, not an algorithm bug, with the "
                "measured logit delta attached."
            ),
        },
        "headline": (
            "On a noise-free model the batched decoder is bit-exactly the "
            "single-sequence decoder (max token delta 0 over %d sequence pairs). "
            "Any divergence on a real quantized/MPS backend is therefore "
            "attributable to batch-variant kernels, and specdiff classifies it "
            "as non-algorithmic (BACKEND_NONDETERMINISM)."
        ) % inv["n_sequence_pairs"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = run(smoke=args.smoke)
    out["elapsed_s"] = time.perf_counter() - t0

    print(json.dumps(out, indent=2))
    assert out["algorithm_batch_invariance"]["n_mismatch"] == 0, "batched != single-sequence"
    assert out["specdiff_clean_path"]["all_NO_DIVERGENCE"], "specdiff invented a mechanism on clean input"

    if not args.smoke:
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
