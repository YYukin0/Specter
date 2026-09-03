"""
P7 Track E verify -- fake-quantized target KV cache x speculative acceptance rate.

Hand-rolled fake-quant (src/kv_fakequant.py): no optimum-quanto / hqq. This is
NOT a packed int-N cache -- it reproduces the numerical error of one, applied to
the target KV, with the draft left in fp16 (independent-draft speculation, the
opposite of QuantSpec-style self-speculation). Question: does a lossy target KV
move the acceptance rate, and in which direction?

Arms: (nbits, kv) in {(16,kv),(8,kv),(4,kv),(8,k_only),(4,k_only)}.
Baseline arm is (16,"kv"); every arm reports alpha_delta vs that baseline.
No pass/fail threshold -- this is a measurement (plan E3).

Real Qwen2.5 0.5B/1.5B on MPS fp16, ~800-token prompts (src/longctx_prompts.py).

Run:
    python src/verify_p7_3_kv_quant.py --smoke   # 2 arms, 2 prompts, no file
    python src/verify_p7_3_kv_quant.py           # full, writes results/p7_3_kv_quant.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kv_fakequant import FakeQuantKVCache
from longctx_prompts import LONGCTX_PROMPTS
from model_loader import load_model_and_tokenizer
from rejection_sampling import collect_eos_ids, encode_prompt
from spec_kv import _cache_position, _new_cache, speculative_generate_kv

DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"
DTYPE = "float16"
SEED_BASE = 1000

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p7_3_kv_quant.json"
HOME = str(Path.home())

GAMMA = 3
MAX_NEW_TOKENS = 48
RESIDUAL_LEN = 64
DRIFT_POSITIONS = 32
ARMS = [(16, "kv"), (8, "kv"), (4, "kv"), (8, "k_only"), (4, "k_only")]


def _make_target_cache(nbits, kv):
    return lambda: FakeQuantKVCache(
        nbits=nbits, residual_len=RESIDUAL_LEN,
        quantize_keys=True, quantize_values=(kv == "kv"))


@torch.no_grad()
def _target_softmax_trace(prompt, target, tok, make_cache, *, n_positions, forced=None):
    """Greedy-decode the target from `prompt` with the given cache factory,
    capturing the full softmax at each step. If `forced` is given, feed those
    tokens instead of the argmax (teacher forcing) so two caches can be compared
    position-for-position. Returns (tokens, [softmax_row, ...])."""
    device = next(target.parameters()).device
    ctx = encode_prompt(tok, prompt, device, True)
    committed = ctx[0].tolist()
    dtype = ctx.dtype
    eos_ids = collect_eos_ids(tok, target)
    cache = make_cache()
    synced = 0
    toks, dists = [], []
    while len(toks) < n_positions:
        pending = committed[synced:]
        feed = torch.tensor([pending], device=device, dtype=dtype)
        pos = _cache_position(synced, len(pending), device)
        logits = target(input_ids=feed, past_key_values=cache, use_cache=True,
                        cache_position=pos).logits
        synced += len(pending)
        row = logits[0, -1, :].float().softmax(-1).cpu()
        dists.append(row)
        nxt = int(forced[len(toks)]) if forced is not None else int(row.argmax())
        toks.append(nxt)
        committed.append(nxt)
        if forced is None and nxt in eos_ids:
            break
    return toks, dists


def _kl(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p / q).log()).sum())


def _run_arm(draft, target, tok, nbits, kv, prompts):
    mc = _make_target_cache(nbits, kv)
    alphas, acc_lens, tps = [], [], []
    kls, quals = [], []
    for i, p in enumerate(prompts):
        seed = SEED_BASE + i
        spec = speculative_generate_kv(
            p, draft, target, tok, gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0, seed=seed, make_cache=mc, make_draft_cache=_new_cache)
        alphas.append(spec.alpha)
        if spec.accept_lengths:
            acc_lens.append(statistics.fmean(spec.accept_lengths))
        tps.append(len(spec.token_ids) / spec.elapsed_s if spec.elapsed_s else 0.0)

        ref_tokens, ref_d = _target_softmax_trace(
            p, target, tok, _new_cache, n_positions=DRIFT_POSITIONS)
        _, q_d = _target_softmax_trace(
            p, target, tok, mc, n_positions=len(ref_tokens), forced=ref_tokens)
        pos_kls = [_kl(q_d[j], ref_d[j]) for j in range(len(ref_tokens))]
        kls.extend(pos_kls)

        q_tokens, _ = _target_softmax_trace(
            p, target, tok, mc, n_positions=len(ref_tokens))
        m = 0
        for a, b in zip(ref_tokens, q_tokens):
            if a != b:
                break
            m += 1
        quals.append(m / max(1, len(ref_tokens)))

    return {
        "nbits": nbits, "kv": kv,
        "alpha_mean": statistics.fmean(alphas) if alphas else 0.0,
        "accept_len_mean": statistics.fmean(acc_lens) if acc_lens else 0.0,
        "tok_per_s_mean": statistics.fmean(tps) if tps else 0.0,
        "kl_mean": statistics.fmean(kls) if kls else 0.0,
        "kl_p90": (statistics.quantiles(kls, n=10)[-1] if len(kls) >= 10 else max(kls, default=0.0)),
        "quality_exact_prefix_ratio": statistics.fmean(quals) if quals else 0.0,
        "n_prompts": len(prompts),
    }


def run(smoke: bool):
    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)

    arms = [(16, "kv"), (4, "kv")] if smoke else ARMS
    prompts = LONGCTX_PROMPTS[:2] if smoke else LONGCTX_PROMPTS

    rows, status = [], "complete"
    t0 = time.perf_counter()
    for nbits, kv in arms:
        row = _run_arm(draft, target, tok, nbits, kv, prompts)
        rows.append(row)
        print(f"[arm nbits={nbits:2d} {kv:7s}] alpha={row['alpha_mean']:.3f} "
              f"acc_len={row['accept_len_mean']:.2f} kl={row['kl_mean']:.4f} "
              f"qual={row['quality_exact_prefix_ratio']:.2f} "
              f"tok/s={row['tok_per_s_mean']:.1f}")
        if not smoke and time.perf_counter() - t0 > 40 * 60:
            status = "partial"
            break

    base = next((r for r in rows if r["nbits"] == 16 and r["kv"] == "kv"), None)
    base_alpha = base["alpha_mean"] if base else 0.0
    for r in rows:
        r["alpha_delta"] = r["alpha_mean"] - base_alpha

    note = ("fake-quant simulates int-N numerical error on the target KV; it does "
            "not pack bytes and saves no memory (echoes note 05's real-int4 "
            "caveat). Direction of alpha_delta is reported as measured, not "
            "assumed. Single machine, greedy, 48 new tokens, long context is "
            "concatenated filler rather than natural long text.")

    out = {
        "task": "P7.3 fake-quant target KV cache x acceptance rate",
        "draft_model": DRAFT_MODEL_NAME, "target_model": TARGET_MODEL_NAME,
        "device": DEVICE, "dtype": DTYPE,
        "gamma": GAMMA, "max_new_tokens": MAX_NEW_TOKENS, "residual_len": RESIDUAL_LEN,
        "drift_positions": DRIFT_POSITIONS,
        "baseline_arm": "(16, kv)",
        "arms": rows,
        "status": status,
        "acceptance_note": note,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = run(args.smoke)
    out["elapsed_s_total"] = time.perf_counter() - t0

    if args.smoke:
        print(json.dumps({"arms": [(r["nbits"], r["kv"], round(r["alpha_delta"], 4),
                                    round(r["kl_mean"], 4)) for r in out["arms"]]}, indent=2))
    else:
        RESULTS_PATH.write_text(json.dumps(out, indent=2).replace(HOME, "~"))
        print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
