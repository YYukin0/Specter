"""
P2.1/P2.2/P2.3 support -- a fixed, self-contained perplexity harness.

Reference: notes/project_plan_v9.md sec 7 P2.1 (perplexity gauge). The absolute
numbers are NOT meant to match the AWQ paper (different model, different window /
stride, different corpus). What matters and is comparable ACROSS our experiments:
  - the perplexity DELTA of self-built 4-bit AWQ vs the fp16 model, and
  - how that delta differs same-distribution vs cross-distribution (P2.2).

Pinned measurement recipe (do not tune between experiments):
  - corpus: wikitext-2-raw-v1 `test` split (natural language) OR mbpp `code`
    fields concatenated (code). NOTE: the plan named codeparrot-clean-valid for
    the code axis, but only its README is in the local HF cache (offline) -- mbpp
    Python solutions are the local stand-in. Documented in load_eval_corpus().
  - join rows with "\n\n", tokenize once WITHOUT special tokens / chat template
    (raw LM perplexity).
  - sliding window: `window` tokens, step `stride`. Non-overlap (stride==window)
    scores window-1 tokens/window; overlap (stride<window) uses the HF recipe --
    only the last `stride` tokens of each window contribute loss, so no token is
    counted twice.
  - loss in fp32 (fp16 cross-entropy on MPS loses precision), torch.no_grad.
  - default window=512, stride=512, seed fixed by the corpus order (deterministic).

perplexity = exp( sum(nll) / n_scored_tokens ).

Run:  python src/awq_perplexity.py --corpus wikitext2 --window 512 --stride 512
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

WIKITEXT_HF = ("Salesforce/wikitext", "wikitext-2-raw-v1")
MBPP_HF = ("google-research-datasets/mbpp", "full")


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #
def load_eval_corpus(name: str, *, max_rows: int | None = None) -> list[str]:
    """name in {"wikitext2", "mbpp_code"} (alias "codeparrot" -> mbpp_code with a
    warning). Returns a list of text rows; the harness joins them. Raises a clear
    error if the local cache does not have the data -- never silently falls back
    to a toy corpus."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    if name == "codeparrot":
        print("WARNING: codeparrot-clean-valid is not in the local HF cache "
              "(only its README). Using mbpp Python solutions as the local code "
              "corpus instead. Set name='mbpp_code' to silence this.", flush=True)
        name = "mbpp_code"

    if name == "wikitext2":
        try:
            ds = load_dataset(WIKITEXT_HF[0], WIKITEXT_HF[1], split="test")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot load wikitext-2 test from local cache: {e}")
        rows = [t for t in ds["text"] if t and t.strip()]
        if max_rows:
            rows = rows[:max_rows]
        if len(rows) < 50:
            raise RuntimeError(f"wikitext-2 test yielded only {len(rows)} non-blank rows")
        return rows

    if name == "mbpp_code":
        try:
            dd = load_dataset(MBPP_HF[0], MBPP_HF[1])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cannot load mbpp from local cache: {e}")
        rows: list[str] = []
        for split in ("test", "train", "validation", "prompt"):
            if split in dd:
                rows += [c for c in dd[split]["code"] if c and c.strip()]
        rows = [r.replace("\r\n", "\n") for r in rows]
        if max_rows:
            rows = rows[:max_rows]
        if len(rows) < 50:
            raise RuntimeError(f"mbpp yielded only {len(rows)} code rows")
        return rows

    raise ValueError(f"unknown corpus {name!r} (want wikitext2 | mbpp_code)")


# --------------------------------------------------------------------------- #
# perplexity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_perplexity(model, tokenizer, texts, *, window=512, stride=512,
                    max_windows=None, join="\n\n") -> dict:
    """Sliding-window perplexity of `model` over `join`.join(texts).

    Returns {"perplexity", "mean_nll", "n_windows", "n_tokens" (scored),
             "n_total_tokens", "window", "stride"}.
    """
    device = next(model.parameters()).device
    enc = tokenizer(join.join(texts), return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    n_total = int(ids.numel())
    if n_total < window + 1:
        raise RuntimeError(f"corpus has {n_total} tokens, need > window+1 ({window+1})")

    nll_sum = torch.zeros((), dtype=torch.float64)
    n_scored = 0
    n_windows = 0
    prev_end = 0
    for begin in range(0, n_total, stride):
        end = min(begin + window, n_total)
        chunk = ids[begin:end].to(device).unsqueeze(0)          # (1, L)
        logits = model(input_ids=chunk).logits[0].float()       # (L, V)
        # position i predicts token i+1
        shift_logits = logits[:-1]                              # (L-1, V)
        shift_labels = chunk[0, 1:]                             # (L-1,)
        # only score tokens whose ABSOLUTE index in `ids` is >= prev_end
        # (the first `prev_end - begin` targets are context carried over)
        first_scored_rel = max(0, prev_end - begin - 1)         # -1: labels are shifted by 1
        lp = F.cross_entropy(shift_logits[first_scored_rel:],
                             shift_labels[first_scored_rel:], reduction="sum")
        k = shift_labels[first_scored_rel:].numel()
        nll_sum += lp.cpu().double()
        n_scored += int(k)
        n_windows += 1
        prev_end = end
        if max_windows and n_windows >= max_windows:
            break
        if end >= n_total:
            break

    mean_nll = float(nll_sum / max(n_scored, 1))
    return {
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
        "mean_nll": mean_nll,
        "n_windows": n_windows,
        "n_tokens": n_scored,
        "n_total_tokens": n_total,
        "window": window,
        "stride": stride,
    }


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--corpus", default="wikitext2", choices=["wikitext2", "mbpp_code", "codeparrot"])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    from model_loader import load_model_and_tokenizer

    texts = load_eval_corpus(args.corpus, max_rows=args.max_rows)
    print(f"corpus {args.corpus}: {len(texts)} rows", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model)
    r = eval_perplexity(model, tokenizer, texts, window=args.window,
                        stride=args.stride, max_windows=args.max_windows)
    print(f"model = {args.model}")
    print(f"perplexity = {r['perplexity']:.4f}  (mean_nll {r['mean_nll']:.4f})")
    print(f"  {r['n_windows']} windows, {r['n_tokens']} scored / {r['n_total_tokens']} total tokens")


if __name__ == "__main__":
    main()
