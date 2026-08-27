"""
P2.1 (first version) -- per-input-channel AWQ scaling search.

Reference: notes/project_plan_v9.md sec 7 P2.1 + appendix A.3; AWQ paper
(Lin et al. 2023, arXiv 2306.00978) Section 3.

Idea (appendix A.3 toy): a Linear's quantization error is not equally costly on
every input channel. Channels whose activations are large carry more of the
output, so we scale their weight column UP by s before quantizing (more effective
quant levels land on it) and scale the matching activation DOWN by 1/s, which is
mathematically identity:  (W * s) @ (x / s) == W @ x. The quant error that
remains sits on the activation side, where the large-magnitude channel is
relatively insensitive to it.

Search (AWQ Section 3, grid form):
  act_scale[j]  = mean_t |x[t, j]|                (from P2.0 stats, per input ch)
  weight_scale[j] = max_i |W[i, j]|              (per input ch, over output rows)
  for alpha in {0, 0.1, ..., 1.0}:
      s = act_scale**alpha / weight_scale**(1 - alpha)      (clamped, then
          renormalised so geometric-mean(s) ~ 1 -- AWQ divides by
          sqrt(s.max() * s.min()) to keep W*s in a sane range)
      loss(alpha) = mean_sq( fakequant(W * s) @ (x / s).T  -  W @ x.T )
  pick alpha with the smallest loss.
alpha = 0  -> scale purely by inverse weight magnitude (ignore activations)
alpha = 1  -> scale purely by activation magnitude (AWQ's pure form)

fake-quant: 4-bit, group-wise (group_size 128), asymmetric, and the order is
  round -> (add zero point) -> CLAMP -> dequantize
Getting this order wrong (e.g. clamp before round) mis-handles values at the
group extremes; test_awq_scaling.py pins it.

This first version does NOT: touch perplexity, compare to GPTQ, do cross-distribution
(P2.2) or calibration-size ablation (P2.3). It only shows: stats feed a scale
search, the search picks an interior alpha, fake-quant is numerically sound, the
toy example checks out.

Run:  python src/awq_scaling.py [--layers N] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from awq_activation_stats import PT_PATH, iter_target_linears  # noqa: E402
from prompts import PROMPTS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "p2_1_scaling_demo.json"

ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
DEFAULT_N_BITS = 4
DEFAULT_GROUP_SIZE = 128


# --------------------------------------------------------------------------- #
# fake quantization
# --------------------------------------------------------------------------- #
def fake_quantize_groupwise(w: torch.Tensor, n_bits: int = DEFAULT_N_BITS,
                            group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Asymmetric group-wise fake quantization of a 2-D weight (out_features,
    in_features). Groups run along the INPUT dim. Returns a dequantized tensor of
    the same shape/dtype -- values snapped to the 4-bit grid but still float.

    Order, deliberately:  q = round(w / scale);  q = q + zero;  q = clamp(q, 0, qmax);
                          w_dq = (q - zero) * scale
    """
    assert w.dim() == 2, "expects (out_features, in_features)"
    out_f, in_f = w.shape
    gs = group_size if (group_size and group_size > 0 and in_f % group_size == 0) else in_f
    wf = w.reshape(out_f, in_f // gs, gs).to(torch.float32)

    qmax = (1 << n_bits) - 1
    max_val = wf.amax(dim=-1, keepdim=True)
    min_val = wf.amin(dim=-1, keepdim=True)
    scale = (max_val - min_val).clamp(min=1e-5) / qmax
    zero = torch.round(-min_val / scale)

    q = torch.round(wf / scale)
    q = q + zero
    q = torch.clamp(q, 0, qmax)          # clamp AFTER round + zero-point
    w_dq = (q - zero) * scale

    return w_dq.reshape(out_f, in_f).to(w.dtype)


def quant_grid_step(w: torch.Tensor, n_bits: int = DEFAULT_N_BITS,
                    group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """The per-group quant step (max-min)/qmax -- handy for error-bound assertions."""
    out_f, in_f = w.shape
    gs = group_size if (group_size and group_size > 0 and in_f % group_size == 0) else in_f
    wf = w.reshape(out_f, in_f // gs, gs).to(torch.float32)
    qmax = (1 << n_bits) - 1
    return (wf.amax(dim=-1, keepdim=True) - wf.amin(dim=-1, keepdim=True)).clamp(min=1e-5) / qmax


# --------------------------------------------------------------------------- #
# scale search
# --------------------------------------------------------------------------- #
def compute_scale(act_scale: torch.Tensor, weight_scale: torch.Tensor,
                  alpha: float) -> torch.Tensor:
    """s[j] = act_scale[j]**alpha / weight_scale[j]**(1-alpha), clamped and
    renormalised so sqrt(max*min) == 1 (keeps W*s from blowing up / vanishing)."""
    a = act_scale.to(torch.float32).clamp(min=1e-6)
    w = weight_scale.to(torch.float32).clamp(min=1e-6)
    s = a.pow(alpha) / w.pow(1.0 - alpha)
    s = s.clamp(min=1e-4)
    s = s / (s.max() * s.min()).sqrt()
    return s


def _out_mse(W: torch.Tensor, X: torch.Tensor, s: torch.Tensor,
             n_bits: int, group_size: int) -> float:
    """mean_sq( fakequant(W*s) @ (X/s).T  -  W @ X.T ), all in fp32."""
    Wf = W.to(torch.float32)
    Xf = X.to(torch.float32)
    ref = Xf @ Wf.T
    Wq = fake_quantize_groupwise(Wf * s, n_bits=n_bits, group_size=group_size)
    approx = (Xf / s) @ Wq.T
    return float(((approx - ref) ** 2).mean())


def search_scale(W: torch.Tensor, X: torch.Tensor, act_scale: torch.Tensor,
                 *, alphas=ALPHA_GRID, n_bits=DEFAULT_N_BITS,
                 group_size=DEFAULT_GROUP_SIZE) -> dict:
    """Grid-search alpha. W: (out, in). X: (n_tokens, in) calibration activations.
    act_scale: (in,) per-input-channel |x| mean from P2.0.

    "no scaling" (s = 1) is an explicit candidate: AWQ keeps the best of the
    searched scales AND the identity, so a layer where the grid can't help is
    left alone rather than made worse (layer 14 mlp.down_proj is such a case)."""
    weight_scale = W.to(torch.float32).abs().amax(dim=0)          # (in,)
    baseline_mse = _out_mse(W, X, torch.ones_like(weight_scale), n_bits, group_size)

    per_alpha = [{"alpha": None, "out_mse": baseline_mse, "s_min": 1.0, "s_max": 1.0, "s_mean": 1.0}]
    for a in alphas:
        s = compute_scale(act_scale, weight_scale, a)
        mse = _out_mse(W, X, s, n_bits, group_size)
        per_alpha.append({"alpha": a, "out_mse": mse,
                          "s_min": float(s.min()), "s_max": float(s.max()),
                          "s_mean": float(s.mean())})
    best = min(per_alpha, key=lambda r: r["out_mse"])
    return {
        "best_alpha": best["alpha"],                               # None == "no scaling wins"
        "baseline_out_mse_no_scaling": baseline_mse,
        "best_out_mse": best["out_mse"],
        "improvement_vs_no_scaling": (1.0 - best["out_mse"] / baseline_mse) if baseline_mse else 0.0,
        "per_alpha": per_alpha,
        "n_bits": n_bits, "group_size": group_size,
        "n_calib_tokens": int(X.shape[0]),
        "in_features": int(W.shape[1]), "out_features": int(W.shape[0]),
    }


# --------------------------------------------------------------------------- #
# driver: capture real layer inputs on a few representative Qwen layers, search
# --------------------------------------------------------------------------- #
def capture_layer_inputs(model, tokenizer, prompts, layer_names, *,
                         max_tokens=2048, max_seq_len=512):
    wanted = set(layer_names)
    buffers = {n: [] for n in layer_names}
    counts = {n: 0 for n in layer_names}
    handles = []

    def mk(name):
        def hook(_m, args):
            if counts[name] >= max_tokens:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.reshape(-1, x.shape[-1]).detach().to(torch.float32).cpu()
            take = min(max_tokens - counts[name], x.shape[0])
            buffers[name].append(x[:take])
            counts[name] += take
        return hook

    for name, mod in iter_target_linears(model):
        if name in wanted:
            handles.append(mod.register_forward_pre_hook(mk(name)))

    device = next(model.parameters()).device
    with torch.no_grad():
        for text in prompts:
            if all(c >= max_tokens for c in counts.values()):
                break
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            model(input_ids=enc["input_ids"].to(device))
    for h in handles:
        h.remove()
    return {n: torch.cat(buffers[n], dim=0) for n in layer_names if buffers[n]}


def _pick_representative_layers(model):
    """One early / one middle / one late block, and within each an attention proj
    and an MLP proj -- enough to see the search behave differently across layer
    types without running all 196."""
    import re
    layer_ids = sorted({int(m.group(1))
                        for n, _ in iter_target_linears(model)
                        if (m := re.search(r"layers\.(\d+)\.", n))})
    if not layer_ids:
        return [n for n, _ in iter_target_linears(model)][:4]
    lo, mid, hi = layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]
    want = []
    for li in (lo, mid, hi):
        want += [f"model.layers.{li}.self_attn.v_proj",
                 f"model.layers.{li}.mlp.down_proj"]
    have = {n for n, _ in iter_target_linears(model)}
    return [w for w in want if w in have]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-calib-tokens", type=int, default=2048)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--smoke", action="store_true", help="2 prompts, 128 tokens, 2 layers")
    args = ap.parse_args()

    from model_loader import load_model_and_tokenizer

    if not PT_PATH.exists():
        sys.exit(f"missing {PT_PATH} -- run src/awq_activation_stats.py first")
    stats = torch.load(PT_PATH)
    act_means = stats["abs_mean"]

    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = _pick_representative_layers(model)
    calib = list(PROMPTS)
    if args.smoke:
        calib = calib[:2]
        args.max_calib_tokens = 128
        args.max_seq_len = 128
        layers = layers[:2]

    print(f"model = {args.model}")
    print(f"representative layers: {layers}")
    print(f"capturing up to {args.max_calib_tokens} calib tokens/layer ...", flush=True)
    X_by_layer = capture_layer_inputs(model, tokenizer, calib, layers,
                                      max_tokens=args.max_calib_tokens,
                                      max_seq_len=args.max_seq_len)

    mod_by_name = {n: m for n, m in iter_target_linears(model)}
    results = {}
    for name in layers:
        if name not in X_by_layer or name not in act_means:
            print(f"  skip {name} (no calib activations / no P2.0 stats)")
            continue
        W = mod_by_name[name].weight.data.detach().float().cpu()  # search runs on CPU (captured X is CPU)
        X = X_by_layer[name]
        act_scale = act_means[name].to(torch.float32)
        r = search_scale(W, X, act_scale)
        # sanity: mathematical identity of the scaling transform (no quant)
        if r["best_alpha"] is None:
            s = torch.ones_like(act_scale)
        else:
            s = compute_scale(act_scale, W.abs().amax(dim=0), r["best_alpha"])
        ident_err = float((((X / s) @ (W * s).T) - (X @ W.T)).abs().max())
        r["scaling_identity_max_abs_err"] = ident_err
        r["act_scale_summary"] = {"min": float(act_scale.min()), "max": float(act_scale.max()),
                                  "median": float(act_scale.median())}
        results[name] = r
        a_str = "none" if r["best_alpha"] is None else f"{r['best_alpha']:.1f}"
        print(f"  {name:<38} best_alpha={a_str:>4}  "
              f"MSE {r['baseline_out_mse_no_scaling']:.3e} -> {r['best_out_mse']:.3e} "
              f"({r['improvement_vs_no_scaling']*100:+.1f}%)  ident_err={ident_err:.1e}", flush=True)

    out = {
        "task": "P2.1 first version -- per-input-channel AWQ scaling search (grid over alpha)",
        "model": args.model,
        "reference": "notes/project_plan_v9.md sec7 P2.1 + appendix A.3; AWQ paper Section 3",
        "fake_quant": {"n_bits": DEFAULT_N_BITS, "group_size": DEFAULT_GROUP_SIZE,
                       "scheme": "asymmetric group-wise", "order": "round -> +zero -> clamp -> dequant"},
        "alpha_grid": ALPHA_GRID,
        "scale_formula": "s = act_scale**alpha / weight_scale**(1-alpha), clamped, "
                         "renormalised by /sqrt(s.max()*s.min())",
        "objective": "mean_sq( fakequant(W*s) @ (X/s).T - W @ X.T ) on captured calibration activations",
        "calibration": {"prompts": "src/prompts.py", "max_tokens_per_layer": args.max_calib_tokens,
                        "max_seq_len": args.max_seq_len},
        "not_done_yet": ["perplexity / end-to-end eval", "GPTQ comparison",
                         "cross-distribution robustness (P2.2)", "calib-size ablation (P2.3)",
                         "full 196-layer sweep", "real int4 packing (cloud, risk B)"],
        "per_layer": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwritten {JSON_PATH.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
