"""
P2.0 -- activation statistics collection for AWQ-style activation-aware quantization.

Reference: notes/project_plan_v9.md sec 7 P2.0 + appendix A.3.

AWQ's premise (Lin et al. 2023, arXiv 2306.00978): weight channels are NOT equally
important. The input channels whose activations have large magnitude carry more
signal, so the corresponding weight columns are "salient" and must be protected
from coarse quantization. To find them we need, per nn.Linear, a per-INPUT-channel
magnitude statistic gathered over a calibration set.

This module only COLLECTS the statistics. The scaling search that uses them is P2.1
(src/awq_scaling.py). No quantization happens here; plain float forward passes.

What is collected, per target Linear, per input channel j:
  abs_mean[j] = mean over all calibration tokens of |x_j|   (AWQ's main signal)
  abs_max[j]  = max  over all calibration tokens of |x_j|    (outlier view; some
                follow-up work keys off the max instead)
Accumulated in float32 (models run fp16 -> fp16 sums overflow / lose precision).

Target layers: the 7 Linear types in a Qwen2 decoder block --
  self_attn.{q,k,v,o}_proj  and  mlp.{gate,up,down}_proj
(lm_head and the embedding are left out: lm_head is not quantized in AWQ's setup
and its input is the final norm output, not a decoder activation.)

Run:  python src/awq_activation_stats.py [--n-calib 24] [--max-seq-len 512] [--smoke]
Writes results/p2_0_activation_stats.pt  (full per-channel vectors)
   and results/p2_0_activation_stats.json (summary + pointer; full vectors are too
   big for JSON -- ~200 layers x up to 8960 channels).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import PROMPTS  # noqa: E402

TARGET_LINEAR_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PT_PATH = RESULTS_DIR / "p2_0_activation_stats.pt"
JSON_PATH = RESULTS_DIR / "p2_0_activation_stats.json"

# A few extra plain-text calibration snippets so the default calib set is ~20-30
# items without pulling in an external corpus (C4 etc. is P2.3's ablation axis).
# Deliberately a mix of prose and code, matching src/prompts.py's split.
_EXTRA_CALIB = [
    "The mitochondria is the powerhouse of the cell, converting nutrients into ATP.",
    "In 1969, Apollo 11 landed the first humans on the Moon after a four-day flight.",
    "A binary search halves the search interval each step, giving O(log n) lookups.",
    "The French Revolution began in 1789 and reshaped European political order.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    p = a[len(a)//2]\n    return quicksort([x for x in a if x < p]) + [x for x in a if x == p] + quicksort([x for x in a if x > p])",
    "Supply and demand determine the equilibrium price in a competitive market.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceans.",
    "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, x):\n        self._items.append(x)\n    def pop(self):\n        return self._items.pop()",
    "Shakespeare wrote 37 plays and 154 sonnets during the Elizabethan era.",
    "TCP guarantees ordered, reliable delivery; UDP trades that for lower latency.",
    "The water cycle moves moisture between oceans, atmosphere, and land continuously.",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 3;",
    "Gravity on the Moon is about one sixth of Earth's surface gravity.",
    "A hash map offers average O(1) insertion and lookup by key.",
    "The Roman Empire reached its greatest territorial extent under Trajan in 117 AD.",
]


def iter_target_linears(model):
    """Yield (qualified_name, module) for every nn.Linear whose leaf name is one of
    TARGET_LINEAR_SUFFIXES."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_LINEAR_SUFFIXES:
            yield name, mod


class ActivationStatsCollector:
    """Forward-PRE-hooks every target Linear and accumulates per-input-channel
    |x| sum and max across every token it sees. Pre-hook (not forward hook) so we
    read the layer INPUT directly -- appendix note in P2.1: hook mounting timing
    (input vs output) is one of the easy-to-get-wrong spots."""

    def __init__(self, model):
        self.model = model
        self._acc = {}          # name -> {"sum_abs", "max_abs", "n_tokens"} or None
        self._handles = []
        self.target_names = []
        for name, mod in iter_target_linears(model):
            self.target_names.append(name)
            self._acc[name] = None
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, args):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)   # (tokens, in_features)
            abs_x = x.abs()
            sum_abs = abs_x.sum(dim=0).detach().cpu()
            max_abs = abs_x.amax(dim=0).detach().cpu()
            n = x.shape[0]
            acc = self._acc[name]
            if acc is None:
                self._acc[name] = {"sum_abs": sum_abs, "max_abs": max_abs, "n_tokens": n}
            else:
                acc["sum_abs"] += sum_abs
                acc["max_abs"] = torch.maximum(acc["max_abs"], max_abs)
                acc["n_tokens"] += n
        return hook

    def result(self):
        """name -> {"abs_mean": tensor[in_features], "abs_max": tensor[in_features],
        "n_tokens": int}. Layers never hit (no calibration data reached them) are
        omitted; `missing()` reports them."""
        out = {}
        for name, acc in self._acc.items():
            if acc is None:
                continue
            out[name] = {
                "abs_mean": acc["sum_abs"] / acc["n_tokens"],
                "abs_max": acc["max_abs"],
                "n_tokens": int(acc["n_tokens"]),
            }
        return out

    def missing(self):
        return [n for n, a in self._acc.items() if a is None]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


def _channel_summary(vec: torch.Tensor) -> dict:
    v = vec.to(torch.float32)
    srt, _ = torch.sort(v, descending=True)
    median = float(v.median())
    return {
        "n_channels": int(v.numel()),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "median": median,
        "p99": float(srt[max(0, int(0.01 * v.numel()) - 1)]),
        # AWQ's whole premise: a few channels dominate. This ratio quantifies it.
        "max_over_median_ratio": (float(v.max()) / median) if median > 0 else float("inf"),
        "top8_channel_idx": torch.topk(v, min(8, v.numel())).indices.tolist(),
    }


def collect(model, tokenizer, prompts, *, max_seq_len=512):
    device = next(model.parameters()).device
    collector = ActivationStatsCollector(model)
    n_seen = 0
    with torch.no_grad(), collector:
        for text in prompts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            model(input_ids=enc["input_ids"].to(device))
            n_seen += 1
    stats = collector.result()
    return stats, {"n_prompts": n_seen, "missing_layers": collector.missing(),
                   "n_target_layers": len(collector.target_names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="the P5.2 target model -- the one that later gets quantized")
    ap.add_argument("--n-calib", type=int, default=24, help="number of calibration prompts")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--smoke", action="store_true", help="2 prompts, seq len 64")
    args = ap.parse_args()

    from model_loader import load_model_and_tokenizer

    calib = list(PROMPTS) + _EXTRA_CALIB
    if args.smoke:
        calib = calib[:2]
        args.max_seq_len = 64
    else:
        calib = calib[:args.n_calib]

    print(f"model = {args.model}")
    print(f"calibration: {len(calib)} prompts, truncated to {args.max_seq_len} tokens", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model)
    stats, meta = collect(model, tokenizer, calib, max_seq_len=args.max_seq_len)

    print(f"collected stats for {len(stats)} / {meta['n_target_layers']} target Linears", flush=True)
    if meta["missing_layers"]:
        print(f"  WARNING missing (never hit): {meta['missing_layers']}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(
        {"abs_mean": {k: v["abs_mean"] for k, v in stats.items()},
         "abs_max": {k: v["abs_max"] for k, v in stats.items()},
         "n_tokens": {k: v["n_tokens"] for k, v in stats.items()},
         "meta": {**meta, "model": args.model, "max_seq_len": args.max_seq_len,
                  "n_calib_prompts": len(calib)}},
        PT_PATH,
    )

    layer_summ = {}
    for name, v in stats.items():
        layer_summ[name] = {
            "n_tokens": v["n_tokens"],
            "abs_mean": _channel_summary(v["abs_mean"]),
            "abs_max": _channel_summary(v["abs_max"]),
        }
    ratios = sorted(((s["abs_mean"]["max_over_median_ratio"], n) for n, s in layer_summ.items()),
                    reverse=True)
    summary = {
        "task": "P2.0 activation statistics collection (AWQ), notes/project_plan_v9.md sec7",
        "model": args.model,
        "calibration": {"n_prompts": len(calib), "max_seq_len": args.max_seq_len,
                        "source": "src/prompts.py (8) + _EXTRA_CALIB prose/code snippets",
                        "note": "calibration-set SIZE and DISTRIBUTION are the P2.3/P2.2 "
                                "ablation axes; this run fixes one small in-distribution set"},
        "target_linear_suffixes": list(TARGET_LINEAR_SUFFIXES),
        "n_target_layers_collected": len(stats),
        "missing_layers": meta["missing_layers"],
        "pt_file": str(PT_PATH.relative_to(RESULTS_DIR.parent)),
        "pt_contents": "abs_mean / abs_max: dict[layer_name -> float32 tensor[in_features]]; n_tokens; meta",
        "salient_channel_evidence": {
            "metric": "max_over_median_ratio of per-channel abs_mean (>> 1 means a few "
                      "input channels dominate -> AWQ scaling has something to protect)",
            "top5_layers": [{"layer": n, "ratio": round(r, 2)} for r, n in ratios[:5]],
            "bottom3_layers": [{"layer": n, "ratio": round(r, 2)} for r, n in ratios[-3:]],
        },
        "per_layer": layer_summ,
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\nsalient-channel ratio (abs_mean max/median), top 5 layers:")
    for r, n in ratios[:5]:
        print(f"  {n:<40} {r:8.1f}")
    print(f"\nwritten {PT_PATH.relative_to(RESULTS_DIR.parent)} + {JSON_PATH.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
