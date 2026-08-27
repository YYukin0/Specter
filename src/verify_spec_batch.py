"""
M5[A] driver -- batched speculative decoding throughput curve, local small batch.

Reference: notes/project_plan_v9.md sec 7 M5[A]; also feeds the P5.3 circuit
breaker's sat_tax anchor (pitfall 15 -- there was no real batch curve before this).

Model pair: main line draft=Qwen2.5-0.5B-Instruct / target=Qwen2.5-1.5B-Instruct
(unchanged). gamma=3 (P1.4 optimum), temperature=1.0, 8 prompts (src/prompts.py),
batch in {1,2,4,8}, seeds {0,1,2}.

No KV cache (see src/spec_batch.py). Reported per batch size (mean +/- std over
seeds):
  - tokens_per_target_forward : total emitted / batched target-forward calls.
    RISES with batch (more sequences resolved per call) -- raw work density.
  - mean_tokens_per_seq_per_round : acceptance efficiency; ~flat across batch is
    expected locally (the math is unchanged and there is no cache contention).
    A drop would be the weak local echo of "speculation gets relatively worse at
    high batch" (which really needs a KV-cache / saturated-GPU setup to show).
  - tok_per_s : wall clock, MPS, NO KV CACHE -> CAVEAT, indicative only.
  - peak_mem_mb : torch.mps.current_allocated_memory() high-water mark; the
    batch->memory slope is the A-track half of the M5 memory record.
  - n_seq_finished_early : sequences that hit EOS before max_new_tokens.

Also does a batch_size=1 vs speculative_generate parity check on the real models
(FakeModel parity is pinned in tests; here we just report the match rate --
tiny MPS fp non-determinism between batched and single forwards can flip a late
token, which is acceptable and noted).

Run:  python src/verify_spec_batch.py [--smoke]
Writes results/p5_5_spec_batch_curve.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer  # noqa: E402
from prompts import PROMPTS  # noqa: E402
from rejection_sampling import speculative_generate  # noqa: E402
from spec_batch import speculative_generate_batch  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "p5_5_spec_batch_curve.json"

GAMMA = 3
TEMPERATURE = 1.0
BATCH_SIZES = [1, 2, 4, 8]
SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 40


def _mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return {"mean": float(t.mean()),
            "std": float(t.std(unbiased=True)) if len(xs) > 1 else 0.0,
            "runs": [float(x) for x in xs]}


def _parity_check(prompts, draft, target, tok, seed, max_new=24):
    """batch_size=1 vs speculative_generate, real models. Return (matched, total).
    Uses a shorter horizon than the main curve -- unit tests already pin the
    FakeModel parity exactly; this is a real-model sanity spot-check."""
    matched = 0
    for p in prompts:
        ref = speculative_generate(p, draft, target, tok, gamma=GAMMA,
                                   max_new_tokens=max_new, temperature=TEMPERATURE,
                                   seed=seed)
        got = speculative_generate_batch([p], draft, target, tok, gamma=GAMMA,
                                         max_new_tokens=max_new, temperature=TEMPERATURE,
                                         seed=seed, batch_size=1)
        if got.token_ids[0] == ref.token_ids:
            matched += 1
    return matched, len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="batch {1,2}, 1 seed, 3 prompts, 16 tokens")
    args = ap.parse_args()

    batch_sizes = BATCH_SIZES
    seeds = SEEDS
    prompts = list(PROMPTS)
    max_new = MAX_NEW_TOKENS
    if args.smoke:
        batch_sizes, seeds, prompts, max_new = [1, 2], [0], list(PROMPTS)[:3], 16

    print(f"loading {DRAFT_MODEL_NAME} + {TARGET_MODEL_NAME} ...", flush=True)
    draft, tok = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, _ = load_model_and_tokenizer(TARGET_MODEL_NAME)

    t0 = time.time()
    print("parity check (batch_size=1 vs speculative_generate) ...", flush=True)
    pm, pt = _parity_check(prompts[:4], draft, target, tok, seed=0)
    print(f"  parity: {pm}/{pt} prompts token-identical", flush=True)

    per_batch = {}
    for bs in batch_sizes:
        runs = {"tokens_per_target_forward": [], "mean_tokens_per_seq_per_round": [],
                "tok_per_s": [], "peak_mem_mb": [], "alpha": [],
                "mean_accept_length": [], "n_seq_finished_early": [],
                "n_target_forwards": [], "wall_s": []}
        for seed in seeds:
            r = speculative_generate_batch(prompts, draft, target, tok, gamma=GAMMA,
                                           max_new_tokens=max_new, temperature=TEMPERATURE,
                                           seed=seed, batch_size=bs)
            runs["tokens_per_target_forward"].append(r.tokens_per_target_forward)
            runs["mean_tokens_per_seq_per_round"].append(r.mean_tokens_per_seq_per_round)
            runs["tok_per_s"].append(r.tok_per_s)
            runs["peak_mem_mb"].append(r.peak_mem_mb)
            runs["alpha"].append(r.alpha)
            runs["mean_accept_length"].append(r.mean_accept_length)
            runs["n_seq_finished_early"].append(r.n_seq_finished_early)
            runs["n_target_forwards"].append(r.n_target_forwards)
            runs["wall_s"].append(r.wall_s)
            print(f"  bs={bs} seed={seed}: tok/tf {r.tokens_per_target_forward:.2f}  "
                  f"tok/seq/round {r.mean_tokens_per_seq_per_round:.3f}  "
                  f"alpha {r.alpha:.3f}  tok/s {r.tok_per_s:.1f}  peak {r.peak_mem_mb:.0f}MB",
                  flush=True)
        per_batch[str(bs)] = {k: _mean_std(v) for k, v in runs.items()}

    # shape read-out
    tf_curve = [(bs, per_batch[str(bs)]["tokens_per_target_forward"]["mean"]) for bs in batch_sizes]
    eff_curve = [(bs, per_batch[str(bs)]["mean_tokens_per_seq_per_round"]["mean"]) for bs in batch_sizes]
    mem_curve = [(bs, per_batch[str(bs)]["peak_mem_mb"]["mean"]) for bs in batch_sizes]
    toks_curve = [(bs, per_batch[str(bs)]["tok_per_s"]["mean"]) for bs in batch_sizes]
    eff_drop = eff_curve[0][1] - eff_curve[-1][1]
    mem_slope = ((mem_curve[-1][1] - mem_curve[0][1]) / (batch_sizes[-1] - batch_sizes[0])
                 if len(batch_sizes) > 1 else 0.0)

    out = {
        "task": "M5[A] batched speculative decoding throughput curve (local small batch)",
        "model_pair": {"draft": DRAFT_MODEL_NAME, "target": TARGET_MODEL_NAME},
        "reference": "notes/project_plan_v9.md sec7 M5[A]; sat_tax anchor for P5.3 (pitfall 15)",
        "config": {"gamma": GAMMA, "temperature": TEMPERATURE, "batch_sizes": batch_sizes,
                   "seeds": seeds, "n_prompts": len(prompts), "max_new_tokens": max_new,
                   "kv_cache": False},
        "ragged_handling": "left-pad + explicit attention_mask + position_ids; per-seq "
                           "gamma clamped by remaining budget; finished seqs stay in the "
                           "padded tensor but are not sampled/scored; round ends when ALL "
                           "seqs finished; one shared generator consumed in seq order "
                           "(so specific tokens depend on batch composition -- still a "
                           "valid draw; batch_size=1 parity is the contract).",
        "parity_batch1_vs_speculative_generate": {"matched": pm, "total": pt,
            "note": "FakeModel parity pinned in tests/test_spec_batch.py; real-model "
                    "mismatches (if any) are late-token flips from MPS fp non-determinism "
                    "between batched and single forwards."},
        "per_batch": per_batch,
        "curves": {
            "tokens_per_target_forward_vs_batch": tf_curve,
            "mean_tokens_per_seq_per_round_vs_batch": eff_curve,
            "peak_mem_mb_vs_batch": mem_curve,
            "tok_per_s_vs_batch_CAVEAT_no_kv_cache": toks_curve,
        },
        "observations": {
            "tokens_per_target_forward_rises_with_batch": tf_curve[-1][1] > tf_curve[0][1],
            "efficiency_drop_bs1_to_bsN": eff_drop,
            "efficiency_drop_is_significant": abs(eff_drop) > 0.3,
            "peak_mem_mb_per_unit_batch": mem_slope,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"\ntokens/target-forward vs batch: {tf_curve}")
    print(f"tok/seq/round vs batch:         {eff_curve}   (drop bs1->bsN = {eff_drop:+.3f})")
    print(f"peak mem MB vs batch:           {mem_curve}   (~{mem_slope:.1f} MB / unit batch)")
    print(f"written {JSON_PATH.relative_to(RESULTS_DIR.parent)}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
