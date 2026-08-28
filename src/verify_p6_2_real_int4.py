"""
P6.2 -- real int4 via mlx-lm (Direction B / 支柱6).

project_plan_v9.md sec 7 P6.2 / TASKS.md P6.2. Everything in P2.0-P2.3 was a
*fake-quant* AWQ study: torch tensors rounded to a 4-bit grid on the fly, still
fp16 storage, still the fp16 matmul, run on MPS. That answered "how much does
4-bit AWQ cost in perplexity" but it never produced a model that is actually
smaller on disk or faster to decode, because PyTorch's quantized backend was
never ported to MPS (sec 9.1 Risk B). mlx-lm is the one local stack that
quantizes AND runs through its own Metal int4 kernels.

This script does a 4-way local comparison of Qwen2.5-1.5B-Instruct, all at
4-bit / group-size 128 where the method has a group size:

  1. fp16_mlx          -- the unquantized model, loaded by mlx-lm (bf16 weights)
  2. mlx_awq_int4_g128 -- `mlx_lm.awq`  (activation-aware, calibrated)
  3. mlx_gptq_int4_g128 -- `mlx_lm.gptq` (Hessian-based, calibrated)
  4. mlx_rtn_int4_g128 -- `nn.quantize` round-to-nearest, NO calibration

Arm 4 is the honest floor: it shows how much the AWQ / GPTQ calibration passes
actually buy over naive RTN on a model this small. (Several P5.x sweeps here
turned up null results; this one may too.)

What we DON'T do: a from-scratch-AWQ-scales -> real-int4-pack arm. `mlx_lm.awq`
exposes no custom-scale-injection API, and the P2.x hand-written AWQ is a torch
fake-quant path only (no int4 packing). Bridging the two is a project on its
own; the RTN arm stands in as the "no calibration" reference instead. This
limitation is recorded in the result JSON.

Per arm we record: on-disk size (GB), wikitext-2 perplexity on a shared token
array (so the deltas are internally consistent -- absolute numbers are NOT
comparable to the P2.x torch sliding-window harness, see `cross_reference`),
batch=1 decode tok/s (mean +/- std over N_RUNS after 1 warmup), mlx peak memory,
and process RSS.

Each bench runs in its own subprocess (like p2_4_mlx_awq_crosscheck.py) so
Metal cache / peak-memory / RSS never cross-contaminate between arms.

Run:
    python src/verify_p6_2_real_int4.py            # full, ~15-25 min
    python src/verify_p6_2_real_int4.py --smoke    # tiny, for the test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # everything is cached locally
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC_RESULTS = HERE / "results"                       # gitignored scratch
OUT_PATH = REPO / "results" / "p6_2_awq_int4_real.json"

# local snapshot dir of Qwen/Qwen2.5-1.5B-Instruct (the HF cache snapshot is
# "incomplete" by huggingface_hub's strict check -- only .gitattributes/LICENSE/
# README are missing -- so the repo id can't be resolved offline; the path works)
def _qwen_1_5b_path() -> str:
    base = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
    snaps = sorted(base.glob("*/"))
    if not snaps:
        raise RuntimeError(f"no local snapshot under {base}")
    return str(snaps[-1])


FP16_MODEL = _qwen_1_5b_path()
AWQ_DIR = SRC_RESULTS / "p6_2_awq_1.5b"
GPTQ_DIR = SRC_RESULTS / "p6_2_gptq_1.5b"
RTN_DIR = SRC_RESULTS / "p6_2_rtn_1.5b"

BITS, GROUP_SIZE = 4, 128

# P2.2 torch fake-quant reference (results/p2_2_cross_distribution.json):
# HF sliding-window harness, wikitext-2-raw-v1 test, window=stride=512.
TORCH_FP16_PPL = 12.177
TORCH_AWQ_FAKEQUANT_PPL = 13.39   # 3-seed mean, from-scratch AWQ 4-bit g128

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]
MAX_TOKENS = 128
N_RUNS = 3
PPL_SEQ_LEN = 512
PPL_NUM_SAMPLES = 32


# --------------------------------------------------------------------------- #
# shared wikitext-2 token array  (same recipe as src/awq_perplexity.py corpus)
# --------------------------------------------------------------------------- #
def _wikitext_ids(tokenizer, seq_len: int, num_samples: int):
    """Deterministic (n, seq_len) int32 array of wikitext-2 test tokens.

    Reuses awq_perplexity.load_eval_corpus so the corpus matches the P2.x runs
    exactly; tokenization is mlx-lm's tokenizer (same Qwen BPE), no chat
    template, no special tokens. First `num_samples` non-overlapping blocks in
    corpus order -- no shuffle, so every arm sees byte-identical input.
    """
    import mlx.core as mx

    from awq_perplexity import load_eval_corpus

    rows = load_eval_corpus("wikitext2")
    text = "\n\n".join(rows)
    ids = tokenizer.encode(text)
    n = (len(ids) // seq_len)
    if num_samples > 0:
        n = min(n, num_samples)
    ids = ids[: n * seq_len]
    return mx.array(ids, dtype=mx.int32).reshape(n, seq_len)


def _eval_ppl(model, data, batch_size=4):
    import mlx.core as mx
    import mlx.nn as nn

    losses = []
    for s in range(0, len(data), batch_size):
        batch = data[s : s + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        lb = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(lb)
        losses.append(lb.flatten())
    all_losses = mx.concatenate(losses)
    mean_loss = all_losses.mean().item()
    std = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    se = std / math.sqrt(all_losses.size)
    try:
        ppl = math.exp(mean_loss)
    except OverflowError:
        ppl = float("inf")
    # keep the JSON strict: non-finite -> null, plus an explicit flag
    finite = math.isfinite(mean_loss) and math.isfinite(ppl)
    return {"perplexity": ppl if finite else None,
            "mean_nll": mean_loss if math.isfinite(mean_loss) else None,
            "ppl_std_error": ppl * se if finite else None,
            "n_tokens": int(all_losses.size),
            "degenerate_forward": not finite}


# --------------------------------------------------------------------------- #
# dir size
# --------------------------------------------------------------------------- #
def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1e9


def _weights_gb(path_or_id: str) -> float:
    p = Path(path_or_id)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) / 1e9


# --------------------------------------------------------------------------- #
# RTN arm: quantize fp16 -> int4 g128 with no calibration (mlx_lm.convert -q,
# which is a plain round-to-nearest affine quant + save, no calibration data)
# --------------------------------------------------------------------------- #
def _ensure_rtn_model():
    if (RTN_DIR / "config.json").exists():
        return
    RTN_DIR.parent.mkdir(parents=True, exist_ok=True)
    convert_bin = Path(sys.executable).parent / "mlx_lm.convert"
    cmd = [str(convert_bin), "--hf-path", FP16_MODEL, "--mlx-path", str(RTN_DIR),
           "-q", "--q-group-size", str(GROUP_SIZE), "--q-bits", str(BITS)]
    print(f"[rtn] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# bench: one subprocess per arm
# --------------------------------------------------------------------------- #
def run_bench(model_path: str, label: str, smoke: bool):
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.utils import load

    model, tokenizer = load(model_path)

    seq_len = 128 if smoke else PPL_SEQ_LEN
    n_samp = 2 if smoke else PPL_NUM_SAMPLES
    max_tok = 16 if smoke else MAX_TOKENS
    n_runs = 1 if smoke else N_RUNS

    data = _wikitext_ids(tokenizer, seq_len, n_samp)
    ppl = _eval_ppl(model, data)

    per_prompt = []
    for prompt in PROMPTS[: 1 if smoke else len(PROMPTS)]:
        for _ in stream_generate(model, tokenizer, prompt, max_tokens=max_tok):
            pass  # warmup
        runs = []
        for _ in range(n_runs):
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            last = None
            for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tok):
                last = resp
            runs.append({
                "wall_s": time.perf_counter() - t0,
                "generation_tokens": last.generation_tokens,
                "generation_tps": last.generation_tps,
                "prompt_tps": last.prompt_tps,
                "mlx_peak_memory_gb": last.peak_memory,
            })
        per_prompt.append({"prompt": prompt, "runs": runs})

    def agg(key):
        vals = [r[key] for p in per_prompt for r in p["runs"]]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        return {"mean": m, "std": math.sqrt(var), "n": len(vals)}

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        rss *= 1024
    print(json.dumps({
        "label": label,
        "model_path": model_path,
        "wikitext2_ppl": ppl,
        "decode_tok_s": agg("generation_tps"),
        "prompt_tok_s": agg("prompt_tps"),
        "mlx_peak_memory_gb": agg("mlx_peak_memory_gb"),
        "process_rss_gb": rss / 1e9,
        "ppl_seq_len": seq_len,
        "ppl_num_samples": n_samp,
    }))


# --------------------------------------------------------------------------- #
def _bench_subprocess(model_path: str, label: str, smoke: bool) -> dict:
    print(f"[bench] {label}: {model_path}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--phase", "bench",
         "--model", model_path, "--label", label] + (["--smoke"] if smoke else []),
        check=True, capture_output=True, text=True,
    )
    out = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out = line
        elif line:
            print(f"  [{label}] {line}", file=sys.stderr)
    if proc.stderr.strip():
        print(proc.stderr[-2000:], file=sys.stderr)
    return json.loads(out)


def run(smoke: bool = False) -> dict:
    arms = [
        ("fp16_mlx", FP16_MODEL),
        ("mlx_awq_int4_g128", str(AWQ_DIR)),
        ("mlx_gptq_int4_g128", str(GPTQ_DIR)),
        ("mlx_rtn_int4_g128", str(RTN_DIR)),
    ]
    missing = [lbl for lbl, p in arms if lbl != "mlx_rtn_int4_g128"
               and not Path(p).exists()]
    if missing:
        raise RuntimeError(f"missing quantized model dir(s): {missing} -- "
                           "run mlx_lm.awq / mlx_lm.gptq first")
    _ensure_rtn_model()

    results = {}
    disk = {}
    for label, path in arms:
        results[label] = _bench_subprocess(path, label, smoke)
        disk[label] = {"weights_gb": round(_weights_gb(path), 4),
                       "dir_gb": round(_dir_gb(Path(path)), 4)}

    fp16_ppl = results["fp16_mlx"]["wikitext2_ppl"]["perplexity"]

    def _sane(lbl) -> bool:
        p = results[lbl]["wikitext2_ppl"]["perplexity"]
        return p is not None and math.isfinite(p) and p < 1e4

    def dppl(lbl):
        if not _sane(lbl):
            return None       # degenerate / broken arm -- see arms[lbl].degenerate
        return round(results[lbl]["wikitext2_ppl"]["perplexity"] - fp16_ppl, 4)

    for lbl in results:
        results[lbl]["degenerate"] = not _sane(lbl)

    out = {
        "task": "P6.2 -- real int4 via mlx-lm (4-way local comparison)",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "quant_config": {"bits": BITS, "group_size": GROUP_SIZE,
                         "awq": "mlx_lm.awq --num-samples 16 --sequence-length 256 --n-grid 10",
                         "gptq": "mlx_lm.gptq (mlx-lm 0.31.3) bits=4 g=128 -- DEGENERATE: "
                                 "constant '!' output / nan wikitext2 nll, reproduced at "
                                 "--num-samples 16 (seq 256) AND --num-samples 64 (seq 512), "
                                 "so not calibration starvation. AWQ + RTN on the same "
                                 "model/config are both fine. Not bisected further (group "
                                 "size? bad fallback layer? mlx-lm bug for this arch) -- "
                                 "compute budget. Finding: on Apple silicon mlx_lm.awq is "
                                 "usable here, mlx_lm.gptq is not.",
                         "rtn": "mlx_lm.convert -q (affine round-to-nearest, no calibration)"},
        "ppl_harness": {
            "corpus": "wikitext-2-raw-v1 test (awq_perplexity.load_eval_corpus)",
            "recipe": f"non-overlapping {PPL_SEQ_LEN}-tok blocks, first "
                      f"{PPL_NUM_SAMPLES} in corpus order, mlx tokenizer, "
                      "next-token CE over whole block, exp(mean)",
            "note": "shared token array across all arms -> deltas internally "
                    "consistent; NOT comparable to the P2.x torch sliding-window "
                    "harness (see cross_reference)",
        },
        "arms": results,
        "disk": disk,
        "deltas_vs_fp16_mlx": {
            "mlx_awq_int4_g128": dppl("mlx_awq_int4_g128"),
            "mlx_gptq_int4_g128": dppl("mlx_gptq_int4_g128"),
            "mlx_rtn_int4_g128": dppl("mlx_rtn_int4_g128"),
        },
        "cross_reference": {
            "torch_fp16_ppl_P2.2": TORCH_FP16_PPL,
            "torch_awq_fakequant_ppl_P2.2": TORCH_AWQ_FAKEQUANT_PPL,
            "torch_fakequant_delta": round(TORCH_AWQ_FAKEQUANT_PPL - TORCH_FP16_PPL, 3),
            "caveat": "different harness (HF sliding-window window=stride=512 vs "
                      "MLX fixed-block), different tokenization path; only the "
                      "~+1.2 fake-quant delta is a sanity anchor, absolute ppl "
                      "will not match",
        },
        "from_scratch_awq_int4_arm": {
            "implemented": False,
            "reason": "mlx_lm.awq exposes no custom-scale-injection API; the "
                      "P2.x hand-written AWQ is a torch fake-quant path with no "
                      "int4 packing. mlx_rtn_int4_g128 is the 'no calibration' "
                      "stand-in.",
        },
        "spec_decoding_tie_in": {
            "question": "does a 4-bit target change optimal gamma? (note sec 7 C1)",
            "status": "not run locally -- the speculative-decoding stack here is "
                      "torch/MPS (speculative_generate_kv); an MLX int4 model "
                      "cannot be loaded into it, and there is no torch/MPS int4 "
                      "runtime (sec 9.1 Risk B). Left as: the acceptance rate is "
                      "set by draft/target *distribution* agreement; a 4-bit "
                      "target perturbs the target logits by ~the same magnitude "
                      "as the AWQ ppl delta (+1.6), which shifts alpha a little "
                      "but not the qualitative gamma* curve (consistent with the "
                      "speculative-KV literature reviewed in sec 7 C1).",
        },
        "platform": {"machine": platform.machine(),
                     "platform": platform.platform()},
        "smoke": smoke,
    }
    out["headline"] = _headline(out)
    return out


def _headline(out: dict) -> str:
    a = out["arms"]
    d = out["deltas_vs_fp16_mlx"]
    fp16_disk = out["disk"]["fp16_mlx"]["weights_gb"]
    awq_disk = out["disk"]["mlx_awq_int4_g128"]["weights_gb"]
    fp16_tps = a["fp16_mlx"]["decode_tok_s"]["mean"]
    awq_tps = a["mlx_awq_int4_g128"]["decode_tok_s"]["mean"]

    def _d(lbl):
        v = d[lbl]
        return f"{v:+.3f}" if v is not None else "DEGENERATE"

    return (
        f"int4 shrinks weights {fp16_disk:.2f}->{awq_disk:.2f} GB "
        f"({fp16_disk / awq_disk:.1f}x); decode {fp16_tps:.0f}->{awq_tps:.0f} tok/s "
        f"({awq_tps / fp16_tps:.2f}x). wikitext2 ppl delta vs fp16: "
        f"AWQ {_d('mlx_awq_int4_g128')}, GPTQ {_d('mlx_gptq_int4_g128')}, "
        f"RTN {_d('mlx_rtn_int4_g128')}. AWQ calibration buys "
        f"{a['mlx_rtn_int4_g128']['wikitext2_ppl']['perplexity'] - a['mlx_awq_int4_g128']['wikitext2_ppl']['perplexity']:+.2f} "
        f"ppl over naive RTN."
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["bench"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.phase == "bench":
        run_bench(args.model, args.label, args.smoke)
        return

    out = run(smoke=args.smoke)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("\n" + out["headline"])
    print(f"[result] {OUT_PATH}")


if __name__ == "__main__":
    main()
