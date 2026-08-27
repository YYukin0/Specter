"""
P2.1 (full version) -- quantize the WHOLE target model with self-built AWQ scaling.

Reference: notes/project_plan_v9.md sec 7 P2.1; AWQ paper (Lin et al. 2023,
arXiv 2306.00978) Section 3. Builds directly on:
  - src/awq_activation_stats.py : per-input-channel |x| stats (P2.0)
  - src/awq_scaling.py          : fake_quantize_groupwise, compute_scale, search_scale

What this adds over the P2.1 first version (which only searched alpha on 6
representative layers): a pipeline that walks every target nn.Linear (196 of them
on Qwen2.5-1.5B-Instruct: q/k/v/o_proj + gate/up/down_proj x 28 blocks), searches
alpha per layer, and writes the fake-quantized weights back IN PLACE so the model
object can then be run through a real perplexity harness (src/awq_perplexity.py).

Design decisions (pinned, do not tune):
  - fake-quant  : 4-bit, group_size 128, asymmetric, round->+zero->clamp->dequant.
                  Reuses fake_quantize_groupwise unchanged (its order is pinned by
                  tests/test_awq_scaling.py).
  - scale search: reuses search_scale -- alpha grid {0.0..1.0} PLUS an explicit
                  "no scaling" (s=1) candidate. Per layer, independent.
  - fall-back semantics: if the search's best candidate is "no scaling"
                  (best_alpha is None), the layer is LEFT UNTOUCHED (stays fp16),
                  NOT quantized with s=1. Rationale: (a) keeps the pipeline
                  honest -- a layer we could not improve is reported as fell_back
                  rather than silently degraded; (b) makes the op idempotent and
                  bit-exact-checkable in tests. The fell_back count is reported
                  prominently in the results JSON because a high count would
                  inflate the post-quant perplexity (those layers keep full
                  precision). On the 6-layer P2.1 demo only 1/6 fell back.
  - fake-quant only: weights stay fp16 floats snapped to the 4-bit grid. No real
                  int4 packing / no real memory saving here -- that is a cloud
                  task (project_plan_v9 risk B). This measures the ACCURACY cost
                  of 4-bit AWQ, which is what P2.2/P2.3 need.

Activation capture: capture_all_layer_inputs() runs ONE forward per calibration
prompt with pre-hooks on all 196 layers at once (not 196 separate forwards),
pools up to `max_tokens_per_layer` token rows per layer on CPU fp32.

Run:  python src/awq_quantize_model.py [--layers-limit N] [--smoke]
Writes results/p2_1_full_quant.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from awq_activation_stats import (  # noqa: E402
    PT_PATH,
    _EXTRA_CALIB,
    iter_target_linears,
)
from awq_scaling import (  # noqa: E402
    DEFAULT_GROUP_SIZE,
    DEFAULT_N_BITS,
    compute_scale,
    fake_quantize_groupwise,
    search_scale,
)
from prompts import PROMPTS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "p2_1_full_quant.json"


# --------------------------------------------------------------------------- #
# activation capture -- all target layers in one pass
# --------------------------------------------------------------------------- #
def capture_all_layer_inputs(model, tokenizer, prompts, *,
                             max_tokens_per_layer=512, max_seq_len=512,
                             apply_chat_template=False):
    """{layer_name -> Tensor(n_tokens<=max_tokens_per_layer, in_features)} on CPU
    fp32. One forward per prompt; every target Linear pre-hooked simultaneously."""
    names = [n for n, _ in iter_target_linears(model)]
    buffers = {n: [] for n in names}
    counts = {n: 0 for n in names}
    handles = []

    def mk(name):
        def hook(_m, args):
            if counts[name] >= max_tokens_per_layer:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.reshape(-1, x.shape[-1]).detach().to(torch.float32).cpu()
            take = min(max_tokens_per_layer - counts[name], x.shape[0])
            if take > 0:
                buffers[name].append(x[:take])
                counts[name] += take
        return hook

    for name, mod in iter_target_linears(model):
        handles.append(mod.register_forward_pre_hook(mk(name)))

    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for text in prompts:
                if all(c >= max_tokens_per_layer for c in counts.values()):
                    break
                if apply_chat_template:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True)
                enc = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_seq_len)
                model(input_ids=enc["input_ids"].to(device))
    finally:
        for h in handles:
            h.remove()
    return {n: torch.cat(buffers[n], dim=0) for n in names if buffers[n]}


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def quantize_model(model, act_stats, calib_inputs_by_layer, *,
                   n_bits=DEFAULT_N_BITS, group_size=DEFAULT_GROUP_SIZE,
                   layers_limit=None, verbose=False,
                   frozen_scales=None, scales_out=None) -> dict:
    """Walk every target Linear, search alpha, write fake-quantized weight in
    place. `act_stats` is the P2.0 dict (needs key "abs_mean": {layer -> tensor}).
    Returns {layer_name -> per-layer record}. Mutates model weights in place.

    Per-layer record keys:
      alpha            : chosen alpha, or None if "no scaling" won (== fell_back)
      fell_back        : True  -> weight left untouched (fp16)
      out_mse_before   : layer-output MSE of plain s=1 fake-quant vs fp16
      out_mse_after    : layer-output MSE of the applied scheme vs fp16
                         (== out_mse_before when fell_back)
      improvement      : 1 - after/before
      s_min/s_max/s_mean : summary of the applied per-channel scale (1.0 when fell_back)
      in_features / out_features / n_calib_tokens
      skipped          : True -> no calib activations or no P2.0 stat, layer untouched

    frozen_scales : {layer_name -> s tensor}. When given for a layer, SKIP the
        alpha search and apply that exact s (None means "leave untouched"). This
        is what makes re-quantization idempotent: the full pipeline is NOT
        idempotent on its own because the alpha search re-reads the (now
        quantized) weights and can land on a slightly different s. Freezing s
        pins it. Used by tests and by anyone re-applying a saved schedule.
    scales_out : optional dict; if given, filled in place with {layer_name -> s
        tensor or None} for every processed layer (so a caller can freeze later).
    """
    act_means = act_stats["abs_mean"]
    records = {}
    targets = list(iter_target_linears(model))
    if layers_limit is not None:
        targets = targets[:layers_limit]

    for i, (name, mod) in enumerate(targets):
        if name not in calib_inputs_by_layer or name not in act_means:
            records[name] = {"skipped": True, "fell_back": False,
                             "reason": "no calib activations" if name not in calib_inputs_by_layer
                             else "no P2.0 stat"}
            if verbose:
                print(f"  [{i+1}/{len(targets)}] {name:<40} SKIP ({records[name]['reason']})",
                      flush=True)
            continue

        orig_dtype = mod.weight.dtype
        orig_device = mod.weight.device
        W = mod.weight.data.detach().to(torch.float32).cpu()          # (out, in)
        X = calib_inputs_by_layer[name].to(torch.float32)             # (n_tok, in)
        act_scale = act_means[name].to(torch.float32)

        frozen = frozen_scales.get(name, "__unset__") if frozen_scales is not None else "__unset__"
        if frozen != "__unset__":
            s = None if frozen is None else frozen.to(torch.float32)
            rec = {
                "skipped": False, "alpha": "frozen" if s is not None else None,
                "fell_back": s is None,
                "out_mse_before": None, "out_mse_after": None, "improvement": None,
                "in_features": int(W.shape[1]), "out_features": int(W.shape[0]),
                "n_calib_tokens": int(X.shape[0]),
                "s_min": 1.0 if s is None else float(s.min()),
                "s_max": 1.0 if s is None else float(s.max()),
                "s_mean": 1.0 if s is None else float(s.mean()),
            }
            if s is not None:
                W_q = fake_quantize_groupwise(W * s, n_bits=n_bits, group_size=group_size) / s
                mod.weight.data = W_q.to(orig_dtype).to(orig_device)
            if scales_out is not None:
                scales_out[name] = s
            records[name] = rec
            if verbose:
                print(f"  [{i+1}/{len(targets)}] {name:<40} FROZEN "
                      f"{'(no scaling)' if s is None else 's applied'}", flush=True)
            continue

        r = search_scale(W, X, act_scale, n_bits=n_bits, group_size=group_size)
        rec = {
            "skipped": False,
            "alpha": r["best_alpha"],
            "fell_back": r["best_alpha"] is None,
            "out_mse_before": r["baseline_out_mse_no_scaling"],
            "out_mse_after": r["best_out_mse"],
            "improvement": r["improvement_vs_no_scaling"],
            "in_features": r["in_features"],
            "out_features": r["out_features"],
            "n_calib_tokens": r["n_calib_tokens"],
            "s_min": 1.0, "s_max": 1.0, "s_mean": 1.0,
        }

        s = None
        if r["best_alpha"] is not None:
            weight_scale = W.abs().amax(dim=0)
            s = compute_scale(act_scale, weight_scale, r["best_alpha"])
            W_q = fake_quantize_groupwise(W * s, n_bits=n_bits, group_size=group_size) / s
            mod.weight.data = W_q.to(orig_dtype).to(orig_device)
            rec["s_min"] = float(s.min())
            rec["s_max"] = float(s.max())
            rec["s_mean"] = float(s.mean())
        if scales_out is not None:
            scales_out[name] = s

        records[name] = rec
        if verbose:
            a = "none" if rec["fell_back"] else f"{rec['alpha']:.1f}"
            print(f"  [{i+1}/{len(targets)}] {name:<40} alpha={a:>4}  "
                  f"MSE {rec['out_mse_before']:.3e}->{rec['out_mse_after']:.3e} "
                  f"({rec['improvement']*100:+.1f}%)", flush=True)

    return records


# --------------------------------------------------------------------------- #
# summary + driver
# --------------------------------------------------------------------------- #
def summarize_records(records: dict) -> dict:
    applied = {n: r for n, r in records.items() if not r.get("skipped") and not r["fell_back"]}
    fell_back = [n for n, r in records.items() if not r.get("skipped") and r["fell_back"]]
    skipped = [n for n, r in records.items() if r.get("skipped")]

    alpha_hist = {}
    for r in applied.values():
        k = f"{r['alpha']:.1f}"
        alpha_hist[k] = alpha_hist.get(k, 0) + 1

    def _dist(xs):
        if not xs:
            return None
        t = torch.tensor(sorted(xs), dtype=torch.float64)
        return {"min": float(t[0]), "max": float(t[-1]), "mean": float(t.mean()),
                "median": float(t.median()), "n": len(xs)}

    by_type = {}
    for n, r in records.items():
        if r.get("skipped"):
            continue
        suffix = n.rsplit(".", 1)[-1]
        d = by_type.setdefault(suffix, {"n": 0, "fell_back": 0, "alphas": []})
        d["n"] += 1
        if r["fell_back"]:
            d["fell_back"] += 1
        else:
            d["alphas"].append(r["alpha"])
    for suffix, d in by_type.items():
        d["alpha_mean"] = float(torch.tensor(d["alphas"]).mean()) if d["alphas"] else None
        d.pop("alphas")

    return {
        "n_target_layers": len(records),
        "n_quantized": len(applied),
        "n_fell_back": len(fell_back),
        "n_skipped": len(skipped),
        "fell_back_layers": sorted(fell_back),
        "skipped_layers": sorted(skipped),
        "alpha_histogram": dict(sorted(alpha_hist.items())),
        "improvement_vs_no_scaling_dist": _dist([r["improvement"] for r in applied.values()]),
        "out_mse_before_dist": _dist([r["out_mse_before"] for r in applied.values()]),
        "out_mse_after_dist": _dist([r["out_mse_after"] for r in applied.values()]),
        "by_layer_type": by_type,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-tokens-per-layer", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--n-calib", type=int, default=24)
    ap.add_argument("--layers-limit", type=int, default=None,
                    help="quantize only the first N target Linears (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 calib prompts, seq 128, first 6 layers")
    args = ap.parse_args()

    from model_loader import load_model_and_tokenizer

    if not PT_PATH.exists():
        sys.exit(f"missing {PT_PATH} -- run src/awq_activation_stats.py first")
    stats = torch.load(PT_PATH)

    calib = list(PROMPTS) + _EXTRA_CALIB
    if args.smoke:
        calib = calib[:2]
        args.max_seq_len = 128
        args.max_tokens_per_layer = 128
        args.layers_limit = 6
    else:
        calib = calib[:args.n_calib]

    print(f"model = {args.model}")
    print(f"calib prompts = {len(calib)}, max_seq_len = {args.max_seq_len}, "
          f"max_tokens/layer = {args.max_tokens_per_layer}, layers_limit = {args.layers_limit}",
          flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model)

    t0 = time.time()
    print("capturing activations for all target layers (1 forward/prompt) ...", flush=True)
    calib_inputs = capture_all_layer_inputs(
        model, tokenizer, calib,
        max_tokens_per_layer=args.max_tokens_per_layer, max_seq_len=args.max_seq_len)
    t_capture = time.time() - t0
    print(f"  captured {len(calib_inputs)} layers in {t_capture:.1f}s", flush=True)

    t1 = time.time()
    print("searching alpha + writing fake-quant weights in place ...", flush=True)
    records = quantize_model(model, stats, calib_inputs,
                             layers_limit=args.layers_limit, verbose=True)
    t_quant = time.time() - t1

    summary = summarize_records(records)
    out = {
        "task": "P2.1 full version -- whole-model self-built AWQ fake-quantization",
        "model": args.model,
        "reference": "notes/project_plan_v9.md sec7 P2.1; AWQ paper Section 3",
        "fake_quant": {"n_bits": DEFAULT_N_BITS, "group_size": DEFAULT_GROUP_SIZE,
                       "scheme": "asymmetric group-wise",
                       "order": "round -> +zero -> clamp -> dequant",
                       "packing": "NONE -- fake-quant only (fp16 floats on 4-bit grid); "
                                  "real int4 packing / memory saving is a cloud task"},
        "fall_back_semantics": "best_alpha=None (no-scaling wins the search) -> layer "
                               "LEFT AT FP16, not quantized. n_fell_back reported below; "
                               "a high count would inflate post-quant perplexity.",
        "alpha_grid": [round(0.1 * i, 1) for i in range(11)],
        "calibration": {"n_prompts": len(calib), "max_seq_len": args.max_seq_len,
                        "max_tokens_per_layer": args.max_tokens_per_layer,
                        "source": "src/prompts.py + awq_activation_stats._EXTRA_CALIB",
                        "note": "calib SIZE is P2.3's axis, calib DISTRIBUTION is P2.2's axis"},
        "timing_s": {"activation_capture": round(t_capture, 1),
                     "alpha_search_and_quant": round(t_quant, 1)},
        "layers_limit": args.layers_limit,
        "summary": summary,
        "per_layer": records,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))

    s = summary
    print(f"\nquantized {s['n_quantized']}/{s['n_target_layers']}  "
          f"fell_back {s['n_fell_back']}  skipped {s['n_skipped']}", flush=True)
    print(f"alpha histogram: {s['alpha_histogram']}", flush=True)
    if s["improvement_vs_no_scaling_dist"]:
        d = s["improvement_vs_no_scaling_dist"]
        print(f"output-MSE improvement vs no-scaling: median {d['median']*100:+.1f}%  "
              f"min {d['min']*100:+.1f}%  max {d['max']*100:+.1f}%", flush=True)
    print(f"written {JSON_PATH.relative_to(RESULTS_DIR.parent)}", flush=True)


if __name__ == "__main__":
    main()
