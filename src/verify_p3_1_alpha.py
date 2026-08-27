"""
P3.1 -- structured / tool-call scenarios vs free-text: does acceptance rate
(alpha) go up when the target is producing structured output?

Reference: notes/project_plan_v9.md §7 P3.1; TASKS.md M3. Model pair is the
main line (draft=Qwen2.5-0.5B-Instruct / target=Qwen2.5-1.5B-Instruct),
gamma=3 (P1.4 optimum), temperature=1.0, seeds {0,1,2} (>=3 reps per §9.6
风险2).

Groups
------
- structured : the `description` of every NON-held-out P3.0 task
  (agentbench_os_tasks.registry.ACTIVE_TASKS). These ask the model to emit
  JSON / file listings / small code edits -- a proxy for "structured /
  tool-call output". Held-out task descriptions are NEVER fed to a model
  (§9.6 风险1); only their pass/fail signal is reserved, but here we don't
  even touch their text.
- freetext : src/prompts.py PROMPTS[:5], the five prose prompts (explain /
  haiku / study tips / summarize / recipe). PROMPTS[5:] are code prompts,
  deliberately excluded from the free-text control.

We measure alpha two ways per group:
  - pooled  = sum(accepted_total) / sum(evaluated_total)  over all runs
  - per_run = mean +/- std of GenResult.alpha across runs

Industry folklore: structured output -> higher acceptance (the draft nails
the boilerplate tokens -- braces, quotes, keywords). Verdict is
"structured_alpha_higher" only if pooled structured - pooled freetext >
0.02 AND the per-run mean gap exceeds one combined std. Otherwise we report
"no clear difference" and note the likely confounds, honestly (§9.6 风险1:
null result is a fine result -- do NOT chase a nicer number).

Reverse check (§9.6 风险4): if structured DOES look higher, before believing
it, we also dump per-item generation length and EOS-early counts, because a
"structured" prompt that just makes the model stop after 5 tokens would
inflate alpha via short easy continuations rather than genuine structural
predictability.

Run:  python src/verify_p3_1_alpha.py [--smoke]
Writes results/p3_1_structured_vs_freetext_alpha.json
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
from agentbench_os_tasks.registry import ACTIVE_TASKS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSON_PATH = RESULTS_DIR / "p3_1_structured_vs_freetext_alpha.json"

GAMMA = 3
TEMPERATURE = 1.0
SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 48

POOLED_GAP_THRESHOLD = 0.02


def _mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=True)) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


def _run_group(name, prompts, draft, target, tok, seeds, *, gamma, temperature, max_new_tokens):
    rows = []
    for i, p in enumerate(prompts):
        for s in seeds:
            r = speculative_generate(
                p, draft, target, tok, gamma=gamma, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=s,
            )
            rows.append({
                "group": name,
                "item": i,
                "prompt_head": p[:90],
                "seed": s,
                "alpha": r.alpha,
                "accepted_total": r.accepted_total,
                "evaluated_total": r.evaluated_total,
                "n_tokens": len(r.token_ids),
                "n_rounds": r.n_rounds,
                "eos_early": len(r.token_ids) < max_new_tokens,
            })
            print(f"  [{name}] item {i} seed {s}: alpha {r.alpha:.3f}  "
                  f"acc {r.accepted_total}/{r.evaluated_total}  "
                  f"toks {len(r.token_ids)}{'  (EOS)' if len(r.token_ids) < max_new_tokens else ''}",
                  flush=True)
    return rows


def _summarize_group(rows):
    acc = sum(r["accepted_total"] for r in rows)
    ev = sum(r["evaluated_total"] for r in rows)
    return {
        "pooled_alpha": (acc / ev) if ev else 0.0,
        "per_run_alpha": _mean_std([r["alpha"] for r in rows]),
        "mean_gen_tokens": _mean_std([r["n_tokens"] for r in rows]),
        "eos_early_frac": sum(r["eos_early"] for r in rows) / len(rows),
        "n_runs": len(rows),
        "accepted_total": acc,
        "evaluated_total": ev,
    }


def _verdict(struct_s, free_s):
    pooled_gap = struct_s["pooled_alpha"] - free_s["pooled_alpha"]
    m_gap = struct_s["per_run_alpha"]["mean"] - free_s["per_run_alpha"]["mean"]
    combined_std = (struct_s["per_run_alpha"]["std"] ** 2 + free_s["per_run_alpha"]["std"] ** 2) ** 0.5
    higher = pooled_gap > POOLED_GAP_THRESHOLD and abs(m_gap) > combined_std
    if higher:
        v = "structured_alpha_higher"
    elif pooled_gap < -POOLED_GAP_THRESHOLD and abs(m_gap) > combined_std:
        v = "structured_alpha_LOWER"
    else:
        v = "no_clear_difference"
    return {
        "verdict": v,
        "pooled_alpha_gap_structured_minus_freetext": pooled_gap,
        "per_run_mean_gap": m_gap,
        "combined_std": combined_std,
        "gap_exceeds_combined_std": abs(m_gap) > combined_std,
        "threshold": POOLED_GAP_THRESHOLD,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3 structured + 3 freetext prompts, 1 seed, 16 tokens")
    args = ap.parse_args()

    seeds = SEEDS
    max_new = MAX_NEW_TOKENS
    structured_prompts = [t.description for t in ACTIVE_TASKS]
    freetext_prompts = list(PROMPTS[:5])
    if args.smoke:
        seeds, max_new = [0], 16
        structured_prompts = structured_prompts[:3]
        freetext_prompts = freetext_prompts[:3]

    print(f"loading {DRAFT_MODEL_NAME} + {TARGET_MODEL_NAME} ...", flush=True)
    draft, tok = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, _ = load_model_and_tokenizer(TARGET_MODEL_NAME)

    t0 = time.time()
    struct_rows = _run_group("structured", structured_prompts, draft, target, tok, seeds,
                             gamma=GAMMA, temperature=TEMPERATURE, max_new_tokens=max_new)
    free_rows = _run_group("freetext", freetext_prompts, draft, target, tok, seeds,
                           gamma=GAMMA, temperature=TEMPERATURE, max_new_tokens=max_new)

    struct_s = _summarize_group(struct_rows)
    free_s = _summarize_group(free_rows)
    verdict = _verdict(struct_s, free_s)

    interpretation = (
        "Structured (P3.0 task descriptions) shows a higher pooled acceptance rate than "
        "free-text prose. Reverse check (§9.6 风险4): compare mean_gen_tokens / "
        "eos_early_frac between groups below -- if the structured group also stops much "
        "earlier, part of the gap may be 'short easy continuation' rather than genuine "
        "structural predictability, not pure structure effect."
        if verdict["verdict"] == "structured_alpha_higher" else
        "No clear acceptance-rate advantage for structured/tool-call-style prompts over "
        "free-text prose at this model pair. Likely reasons: (a) the P3.0 task "
        "*descriptions* are themselves English prose -- the model's *completion* is only "
        "partly structured within a 48-token window; (b) the 0.5B/1.5B pair may already "
        "agree on boilerplate tokens regardless of domain, leaving little headroom; "
        "(c) temperature=1.0 adds noise that swamps a small real effect. Reported as a "
        "null result per §9.6 风险1; not chasing a nicer number by cherry-picking prompts "
        "or seeds."
    )

    out = {
        "task": "P3.1 structured/tool-call vs free-text acceptance rate (alpha)",
        "model_pair": {"draft": DRAFT_MODEL_NAME, "target": TARGET_MODEL_NAME},
        "reference": "notes/project_plan_v9.md §7 P3.1; TASKS.md M3",
        "config": {
            "gamma": GAMMA, "temperature": TEMPERATURE, "seeds": seeds,
            "max_new_tokens": max_new,
            "n_structured_prompts": len(structured_prompts),
            "n_freetext_prompts": len(freetext_prompts),
            "structured_source": "agentbench_os_tasks.registry.ACTIVE_TASKS[*].description "
                                 "(non-held-out only; held-out task text never fed to a model)",
            "freetext_source": "src/prompts.py PROMPTS[:5] (prose only; code prompts [5:] excluded)",
        },
        "structured": struct_s,
        "freetext": free_s,
        "comparison": verdict,
        "interpretation": interpretation,
        "per_run_rows": struct_rows + free_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nstructured pooled alpha {struct_s['pooled_alpha']:.4f}  "
          f"(per-run {struct_s['per_run_alpha']['mean']:.4f} +/- {struct_s['per_run_alpha']['std']:.4f})")
    print(f"freetext   pooled alpha {free_s['pooled_alpha']:.4f}  "
          f"(per-run {free_s['per_run_alpha']['mean']:.4f} +/- {free_s['per_run_alpha']['std']:.4f})")
    print(f"verdict: {verdict['verdict']}  (pooled gap {verdict['pooled_alpha_gap_structured_minus_freetext']:+.4f})")
    print(f"written {JSON_PATH.relative_to(RESULTS_DIR.parent)}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
