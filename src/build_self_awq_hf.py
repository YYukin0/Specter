"""
支柱7 Bullet 3 -- materialise the self-built AWQ model as an HF checkpoint so it
can be served by an OpenAI-compatible endpoint (mlx-lm) and evaluated with the
EleutherAI lm-evaluation-harness alongside the fp16 baseline and the mlx-lm
production int4 model.

This does NOT introduce a new quantiser. It runs the exact P2.1/P2.2 pipeline
(`capture_all_layer_inputs` -> `quantize_model`, 4-bit group-128 asymmetric
fake-quant, per-layer alpha search, fp16 fall-back for layers that don't improve)
and writes the fake-quantised weights out as a normal HF safetensors checkpoint.
The weights are fp16 floats snapped to the 4-bit grid, so served at fp16 the
downstream accuracy is exactly the fake-quant accuracy P2.2 measured on
perplexity -- which is the whole point: does the ppl hit translate to a GSM8K /
IFEval hit, and by how much.

Calibration is pinned to the P2.2 primary cell: calib = wikitext-2 (NL),
n_calib = 32 rows, shuffle seed 0, max 512 tokens/row.

Run:  python src/build_self_awq_hf.py [--out DIR] [--smoke]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from awq_perplexity import load_eval_corpus  # noqa: E402
from awq_quantize_model import (  # noqa: E402
    capture_all_layer_inputs,
    quantize_model,
    summarize_records,
)
from model_loader import load_model_and_tokenizer  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_CALIB = 32
CALIB_SEED = 0
CALIB_MAX_SEQ_LEN = 512
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "bullet3_self_awq_1.5b_hf"


def _stats_from_capture(calib_inputs):
    return {"abs_mean": {n: X.abs().mean(dim=0) for n, X in calib_inputs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="n_calib=6, first 12 layers -- plumbing check only")
    args = ap.parse_args()

    n_calib = 6 if args.smoke else N_CALIB
    layers_limit = 12 if args.smoke else None

    t0 = time.time()
    print("loading wikitext-2 calibration pool ...", flush=True)
    nl_pool = load_eval_corpus("wikitext2")
    rng = random.Random(CALIB_SEED)
    shuffled = nl_pool[:]
    rng.shuffle(shuffled)
    calib_texts = shuffled[:n_calib]
    print(f"  {len(calib_texts)} calib rows (seed {CALIB_SEED})", flush=True)

    model, tokenizer = load_model_and_tokenizer(MODEL)

    print("capturing calibration activations ...", flush=True)
    calib_inputs = capture_all_layer_inputs(
        model, tokenizer, calib_texts,
        max_tokens_per_layer=CALIB_MAX_SEQ_LEN, max_seq_len=CALIB_MAX_SEQ_LEN)
    stats = _stats_from_capture(calib_inputs)

    print("running self-built AWQ (per-layer alpha search) ...", flush=True)
    recs = quantize_model(model, stats, calib_inputs, layers_limit=layers_limit,
                          verbose=True)
    summ = summarize_records(recs)
    print(f"  quantized {summ['n_quantized']}  fell_back {summ['n_fell_back']}  "
          f"skipped {summ['n_skipped']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"writing HF checkpoint -> {args.out} ...", flush=True)
    model = model.to("cpu")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    meta = {
        "task": "支柱7 Bullet 3 -- self-built AWQ materialised as HF checkpoint",
        "base_model": MODEL,
        "quant": "4-bit group-128 asymmetric fake-quant (awq_scaling.fake_quantize_groupwise), "
                 "per-layer alpha search, fp16 fall-back for non-improving layers",
        "served_as": "fp16 (weights are fp16 floats on the 4-bit grid) -- downstream "
                     "accuracy == fake-quant accuracy",
        "calibration": {"corpus": "wikitext-2 (NL)", "n_calib_rows": n_calib,
                        "shuffle_seed": CALIB_SEED, "max_seq_len": CALIB_MAX_SEQ_LEN},
        "layer_summary": {k: summ[k] for k in
                          ("n_quantized", "n_fell_back", "n_skipped",
                           "alpha_histogram", "fell_back_layers")
                          if k in summ},
        "smoke": args.smoke,
        "build_seconds": round(time.time() - t0, 1),
    }
    (args.out / "specter_build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\ndone in {meta['build_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
