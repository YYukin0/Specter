"""
P1.0 mlx-lm cross-check — see notes/project_plan_v9.md §7 P1.0 and §9.6.

Not a replacement for gate_p1_0.py: this exists to sanity-check that the
alpha measured by the hand-rolled HF/PyTorch/MPS harness is not an artifact
of that harness. Runs the same model pair (mlx-community bf16 conversions,
so precision matches the HF harness rather than confounding "different
framework" with "different precision") through mlx-lm's built-in
speculative decoding and derives an alpha estimate from its public
stream_generate API.

mlx-lm's `from_draft` flag on each yielded token is True only when that
token was accepted unmodified from the draft model. Every round drafts
exactly `num_draft_tokens` candidates and ends in exactly one token that did
NOT come from the draft model (either a rejection-resample or the bonus
token) -- so:
    num_rounds   = count(from_draft == False)
    num_proposed = num_rounds * num_draft_tokens   (approx; last round may be short)
    alpha        = count(from_draft == True) / num_proposed
"""
import json
from pathlib import Path

from mlx_lm import load, stream_generate

from prompts import PROMPTS

DRAFT_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
TARGET_MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-bf16"
NUM_DRAFT_TOKENS = 4
MAX_TOKENS = 40
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p1_0_mlx_crosscheck_result.json"


def run_prompt(model, tokenizer, draft_model, prompt, num_draft_tokens, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    text_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    accepted = 0
    total = 0
    non_draft_events = 0
    for response in stream_generate(
        model,
        tokenizer,
        text_prompt,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        max_tokens=max_tokens,
    ):
        total += 1
        if response.from_draft:
            accepted += 1
        else:
            non_draft_events += 1

    num_proposed = non_draft_events * num_draft_tokens
    alpha = accepted / num_proposed if num_proposed else 0.0
    return {
        "prompt": prompt,
        "tokens_generated": total,
        "accepted_from_draft": accepted,
        "non_draft_events": non_draft_events,
        "num_proposed_estimate": num_proposed,
        "alpha": alpha,
    }


def main():
    print(f"draft  = {DRAFT_MODEL_ID}")
    print(f"target = {TARGET_MODEL_ID}")
    print(f"num_draft_tokens = {NUM_DRAFT_TOKENS}")
    print()

    print("loading models...")
    target_model, tokenizer = load(TARGET_MODEL_ID)
    draft_model, draft_tokenizer = load(DRAFT_MODEL_ID)

    if draft_tokenizer.vocab_size != tokenizer.vocab_size:
        result = {
            "draft_model": DRAFT_MODEL_ID,
            "target_model": TARGET_MODEL_ID,
            "vocab_match": False,
            "conclusion": "mlx-lm's own vocab_size check failed -- consistent with the HF harness gate on vocab mismatch.",
        }
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    per_prompt_results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i + 1}/{len(PROMPTS)}] {prompt[:60]!r}")
        r = run_prompt(target_model, tokenizer, draft_model, prompt, NUM_DRAFT_TOKENS, MAX_TOKENS)
        print(f"    accepted {r['accepted_from_draft']}/{r['num_proposed_estimate']}  alpha={r['alpha']:.3f}")
        per_prompt_results.append(r)

    total_accepted = sum(r["accepted_from_draft"] for r in per_prompt_results)
    total_proposed = sum(r["num_proposed_estimate"] for r in per_prompt_results)
    overall_alpha = total_accepted / total_proposed if total_proposed else 0.0

    result = {
        "draft_model": DRAFT_MODEL_ID,
        "target_model": TARGET_MODEL_ID,
        "num_draft_tokens": NUM_DRAFT_TOKENS,
        "vocab_match": True,
        "per_prompt": per_prompt_results,
        "total_accepted": total_accepted,
        "total_proposed_estimate": total_proposed,
        "overall_alpha": overall_alpha,
        "note": "alpha here is derived from mlx-lm's from_draft flag, not directly comparable bit-for-bit "
                "to the HF greedy-verification harness (different framework, same bf16 precision) -- "
                "it is a magnitude sanity check only, per project_plan_v9.md §9.6.",
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print()
    print("=" * 60)
    print(f"mlx-lm cross-check alpha = {overall_alpha:.4f}  ({total_accepted}/{total_proposed})")
    print("=" * 60)


if __name__ == "__main__":
    main()
