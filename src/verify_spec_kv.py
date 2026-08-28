"""
P6.0 verify -- the project's first citable real speedup number.

Runs the real Qwen2.5 0.5B/1.5B pair on MPS fp16 and compares four decoders on
identical prompts / seeds:

  * speculative_generate_kv   -- P6.0, KV cache on both models
  * speculative_generate      -- the old no-cache path (re-reads the whole prefix
                                 gamma times per round)
  * target_only_generate_kv   -- the FAIR baseline (KV-cached target only)
  * target_only_generate      -- the old no-cache baseline

Reported per gamma in {1,3,5}, seeds {0,1,2}, over the 8 P1.x prompts:
  * tok/s mean +/- std for every decoder
  * speedup of spec-KV vs BOTH target-only baselines
  * draft work: total tokens fed to the draft model per generated token -- should
    collapse from "~gamma * prefix" (no cache) to "~gamma * 1" (KV)
  * real-model parity: the common-prefix fraction between spec-KV and the
    no-cache spec path. MPS fp16 reductions are not deterministic, so this is
    "long common prefix", not bit-exact -- same caveat as src/spec_batch.py and
    the EQSPEC/EXSPEC literature (~95% exact). The bit-exact contract lives in
    tests/test_spec_kv.py on the deterministic FakeModel.

Run:  python src/verify_spec_kv.py            (full: ~a few min on the 24GB Mac)
      python src/verify_spec_kv.py --smoke    (gamma=3, seed 0, 3 prompts)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from prompts import PROMPTS
from rejection_sampling import speculative_generate, target_only_generate
from spec_kv import speculative_generate_kv, target_only_generate_kv

GAMMAS = [1, 3, 5]
SEEDS = [0, 1, 2]
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 64
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p6_0_kv_cache_speculative.json"


def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def _common_prefix_frac(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / n


def _tps(result):
    return len(result.token_ids) / result.elapsed_s if result.elapsed_s else 0.0


def greedy_parity(prompts, gammas, draft_model, target_model, tokenizer):
    """Real-model correctness probe. Greedy (temperature=0) only argmax matters,
    so MPS fp16 noise almost never flips a token -- this is where the literature's
    "~exact match" lives. Reports the common-prefix fraction of spec-KV against
    both the no-cache spec path and KV-cached target-only."""
    out = []
    for gamma in gammas:
        vs_nc, vs_to = [], []
        for prompt in prompts:
            kv = speculative_generate_kv(prompt, draft_model, target_model, tokenizer,
                                         gamma=gamma, max_new_tokens=48,
                                         temperature=0.0, seed=0)
            nc = speculative_generate(prompt, draft_model, target_model, tokenizer,
                                      gamma=gamma, max_new_tokens=48,
                                      temperature=0.0, seed=0)
            to = target_only_generate_kv(prompt, target_model, tokenizer,
                                         max_new_tokens=48, temperature=0.0, seed=0)
            vs_nc.append(_common_prefix_frac(kv.token_ids, nc.token_ids))
            vs_to.append(_common_prefix_frac(kv.token_ids, to.token_ids))
        out.append({
            "gamma": gamma,
            "spec_kv_vs_spec_nocache_prefix_frac_mean": _mean_std(vs_nc)[0],
            "spec_kv_vs_spec_nocache_prefix_frac_min": min(vs_nc),
            "spec_kv_vs_target_only_kv_prefix_frac_mean": _mean_std(vs_to)[0],
            "spec_kv_vs_target_only_kv_prefix_frac_min": min(vs_to),
            "n_prompts_exact": sum(1 for x in vs_nc if x == 1.0),
        })
    return out


def run(prompts, gammas, seeds):
    draft_model, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target_model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    # --- target-only baselines (gamma-independent) ------------------------------
    base_kv, base_nc = [], []
    for seed in seeds:
        for prompt in prompts:
            r_kv = target_only_generate_kv(prompt, target_model, tokenizer,
                                           max_new_tokens=MAX_NEW_TOKENS,
                                           temperature=TEMPERATURE, seed=seed)
            r_nc = target_only_generate(prompt, target_model, tokenizer,
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        temperature=TEMPERATURE, seed=seed)
            base_kv.append(_tps(r_kv))
            base_nc.append(_tps(r_nc))
    base_kv_m, base_kv_s = _mean_std(base_kv)
    base_nc_m, base_nc_s = _mean_std(base_nc)

    greedy = greedy_parity(prompts, gammas, draft_model, target_model, tokenizer)

    rows = []
    for gamma in gammas:
        kv_tps, nc_tps = [], []
        kv_draft_tok_per_gen, nc_draft_tok_per_gen = [], []
        prefix_fracs = []
        kv_alpha, kv_accept_mean = [], []
        for seed in seeds:
            for prompt in prompts:
                kv = speculative_generate_kv(prompt, draft_model, target_model, tokenizer,
                                             gamma=gamma, max_new_tokens=MAX_NEW_TOKENS,
                                             temperature=TEMPERATURE, seed=seed)
                nc = speculative_generate(prompt, draft_model, target_model, tokenizer,
                                          gamma=gamma, max_new_tokens=MAX_NEW_TOKENS,
                                          temperature=TEMPERATURE, seed=seed)
                kv_tps.append(_tps(kv))
                nc_tps.append(_tps(nc))
                if kv.token_ids:
                    kv_draft_tok_per_gen.append(kv.draft_forward_tokens / len(kv.token_ids))
                # no-cache draft work: gamma autoregressive passes over a prefix
                # that grows each round; approximate as gamma * mean_prefix_len
                approx_prefix = len(tokenizer(prompt)["input_ids"]) + len(nc.token_ids) / 2
                if nc.token_ids:
                    nc_draft_tok_per_gen.append(gamma * approx_prefix / len(nc.token_ids) * len(nc.accept_lengths))
                prefix_fracs.append(_common_prefix_frac(kv.token_ids, nc.token_ids))
                kv_alpha.append(kv.alpha)
                kv_accept_mean.append(statistics.fmean(kv.accept_lengths) if kv.accept_lengths else 0.0)

        kv_m, kv_s = _mean_std(kv_tps)
        nc_m, nc_s = _mean_std(nc_tps)
        pf_m, pf_s = _mean_std(prefix_fracs)
        rows.append({
            "gamma": gamma,
            "spec_kv_tok_per_s_mean": kv_m,
            "spec_kv_tok_per_s_std": kv_s,
            "spec_nocache_tok_per_s_mean": nc_m,
            "spec_nocache_tok_per_s_std": nc_s,
            "speedup_spec_kv_vs_target_only_kv": (kv_m / base_kv_m) if base_kv_m else 0.0,
            "speedup_spec_kv_vs_target_only_nocache": (kv_m / base_nc_m) if base_nc_m else 0.0,
            "speedup_spec_kv_vs_spec_nocache": (kv_m / nc_m) if nc_m else 0.0,
            "draft_tokens_fed_per_generated_token_kv": _mean_std(kv_draft_tok_per_gen)[0],
            "draft_tokens_fed_per_generated_token_nocache_approx": _mean_std(nc_draft_tok_per_gen)[0],
            "sampling_mode_common_prefix_frac_mean": pf_m,
            "sampling_mode_common_prefix_frac_min": min(prefix_fracs) if prefix_fracs else 1.0,
            "sampling_mode_common_prefix_frac_std": pf_s,
            "spec_kv_alpha_mean": _mean_std(kv_alpha)[0],
            "spec_kv_accept_length_mean": _mean_std(kv_accept_mean)[0],
        })

    return {
        "task": "P6.0 KV-cache-correct single-sequence speculative decoding",
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "device": "mps",
        "dtype": "float16",
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gammas": gammas,
        "seeds": seeds,
        "n_prompts": len(prompts),
        "target_only_kv_tok_per_s_mean": base_kv_m,
        "target_only_kv_tok_per_s_std": base_kv_s,
        "target_only_nocache_tok_per_s_mean": base_nc_m,
        "target_only_nocache_tok_per_s_std": base_nc_s,
        "kv_cache_speedup_target_only": (base_kv_m / base_nc_m) if base_nc_m else 0.0,
        "greedy_parity_real_model": greedy,
        "per_gamma": rows,
        "caveats": [
            "Wall-clock on MPS fp16. This is the project's first real (KV-cached) "
            "speedup number; still a single 24GB Mac, single sequence.",
            "greedy_parity_real_model is the real correctness signal: greedy only "
            "depends on argmax so fp noise rarely flips a token (expect ~1.0). The "
            "per_gamma sampling_mode_common_prefix_frac is much lower (~0.7) purely "
            "because temperature=1.0 multinomial draws cascade on any 1e-6 fp diff "
            "-- expected, not a bug. Bit-exact parity is pinned on the FakeModel in "
            "tests/test_spec_kv.py.",
            "draft_tokens_fed_per_generated_token_nocache_approx is an estimate "
            "(gamma passes over a linearly growing prefix); the KV figure is exact.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        prompts, gammas, seeds = PROMPTS[:3], [3], [0]
    else:
        prompts, gammas, seeds = PROMPTS, GAMMAS, SEEDS

    t0 = time.perf_counter()
    out = run(prompts, gammas, seeds)
    out["elapsed_s_total"] = time.perf_counter() - t0

    if not args.smoke:
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(f"wrote {RESULTS_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
