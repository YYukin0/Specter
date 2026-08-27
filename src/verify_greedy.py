"""
P1.2 -- greedy-mode correctness verifier for the P1.1 speculative decoder.

Two halves, both required (notes/project_plan_v9.md sec 7 P1.2 + sec 9.6 risk 3):

  1. FORWARD check  -- in greedy mode (temperature == 0), the speculative decoder
     must emit exactly the same token sequence as plain target-only greedy
     decoding, token for token. Any divergence is reported with its position and
     the two token strings (float16 on MPS can in principle accumulate enough
     error to flip an argmax; we want the specifics, not a hand-wave).

  2. REVERSE check (fault injection) -- deliberately break the P1.1 implementation
     with two known-bad bugs and confirm the forward check *catches* them (the
     emitted sequence diverges from target-only greedy). A verifier that cannot
     be shown to detect a planted bug is worse than no verifier (sec 9.6 risk 3).
       bug A: bonus token sampled from the DRAFT model (坑2)
       bug B: force-accept the draft token at within-round index 0

The injected bugs live only in `Injection` (src/rejection_sampling.py) and are off
by default; nothing here mutates the production decode path.

Run:  python src/verify_greedy.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from prompts import PROMPTS
from rejection_sampling import (
    Injection,
    encode_prompt,
    speculative_generate,
    target_only_generate,
)

GAMMA = 4
MAX_NEW_TOKENS = 64
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p1_2_greedy_verifier.json"


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


@torch.no_grad()
def hf_greedy_reference(prompt, target_model, tokenizer):
    """Independent third implementation: HF `generate`, greedy, with all of
    Qwen2.5's sampling / repetition-penalty knobs explicitly neutralised so it is
    a true raw-argmax decode (the default generation_config carries
    repetition_penalty=1.1, which would otherwise diverge from raw argmax)."""
    device = next(target_model.parameters()).device
    ids = encode_prompt(tokenizer, prompt, device, apply_chat_template=True)
    out = target_model.generate(
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return out[0, ids.shape[1]:].tolist()


def forward_check(draft_model, target_model, tokenizer):
    rows = []
    for i, prompt in enumerate(PROMPTS):
        spec = speculative_generate(
            prompt, draft_model, target_model, tokenizer,
            gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
        )
        ref = target_only_generate(
            prompt, target_model, tokenizer, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
        )
        hf_ref = hf_greedy_reference(prompt, target_model, tokenizer)

        # Compare on the overlapping prefix. Speculative decoding emits the accepted
        # prefix + 1 token per round, so its final round can overshoot max_new_tokens
        # by up to gamma-1 tokens; that overshoot is not a correctness difference.
        # A genuine difference is either a token mismatch within the common prefix
        # or one run stopping early on EOS while the other kept going.
        n = min(len(spec.token_ids), len(ref.token_ids))
        div = _first_divergence(spec.token_ids[:n], ref.token_ids[:n])
        hf_n = min(len(spec.token_ids), len(hf_ref))
        hf_div = _first_divergence(spec.token_ids[:hf_n], hf_ref[:hf_n])
        spec_eos_short = len(spec.token_ids) < MAX_NEW_TOKENS  # stopped on EOS
        ref_eos_short = len(ref.token_ids) < MAX_NEW_TOKENS
        eos_mismatch = spec_eos_short != ref_eos_short

        row = {
            "prompt": prompt,
            "spec_len": len(spec.token_ids),
            "ref_len": len(ref.token_ids),
            "hf_ref_len": len(hf_ref),
            "match_vs_target_only": div is None and not eos_mismatch,
            "match_vs_hf_generate": hf_div is None,
            "first_divergence_index": div,
            "mean_accept_len": (sum(spec.accept_lengths) / len(spec.accept_lengths))
            if spec.accept_lengths else 0.0,
        }
        if div is not None:
            row["divergence_detail"] = {
                "index": div,
                "spec_token": tokenizer.decode([spec.token_ids[div]]) if div < len(spec.token_ids) else None,
                "ref_token": tokenizer.decode([ref.token_ids[div]]) if div < len(ref.token_ids) else None,
            }
        rows.append(row)
        status = "OK " if row["match_vs_target_only"] else "DIFF"
        print(f"  [{i+1}/{len(PROMPTS)}] {status}  accept_len~{row['mean_accept_len']:.2f}  {prompt[:48]!r}")
    return rows


def reverse_check(draft_model, target_model, tokenizer):
    """Plant each bug, confirm the forward check would flag it (spec != target-only)."""
    bugs = {
        "bonus_from_draft": Injection(bonus_from_draft=True),
        "force_accept_index_0": Injection(force_accept_index=0),
    }
    out = {}
    for name, inj in bugs.items():
        detected = 0
        opportunities = 0
        details = []
        for prompt in PROMPTS:
            ref = target_only_generate(
                prompt, target_model, tokenizer, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
            )
            bad = speculative_generate(
                prompt, draft_model, target_model, tokenizer,
                gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, injection=inj,
            )
            # opportunity = the buggy branch actually had a chance to fire
            if name == "bonus_from_draft":
                opp = any(n == GAMMA for n in bad.accept_lengths)
            else:
                opp = any(n == 0 for n in bad.accept_lengths) or True  # index 0 is tested every round
            opportunities += int(opp)
            n = min(len(bad.token_ids), len(ref.token_ids))
            div = _first_divergence(bad.token_ids[:n], ref.token_ids[:n])
            eos_mismatch = (len(bad.token_ids) < MAX_NEW_TOKENS) != (len(ref.token_ids) < MAX_NEW_TOKENS)
            diverged = div is not None or eos_mismatch
            detected += int(diverged)
            details.append({"prompt": prompt[:48], "opportunity": bool(opp),
                            "diverged_from_target_only": bool(diverged),
                            "first_divergence_index": div})
        passed = detected > 0
        out[name] = {
            "prompts_with_opportunity": opportunities,
            "prompts_where_verifier_flagged": detected,
            "verifier_catches_this_bug": passed,
            "per_prompt": details,
        }
        print(f"  bug '{name}': flagged on {detected}/{len(PROMPTS)} prompts -> "
              f"{'PASS (verifier is not blind)' if passed else 'FAIL (verifier blind!)'}")
    return out


def main():
    print(f"draft  = {DRAFT_MODEL_NAME}")
    print(f"target = {TARGET_MODEL_NAME}")
    print(f"gamma  = {GAMMA}, max_new_tokens = {MAX_NEW_TOKENS}\n")

    draft_model, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target_model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)
    assert draft_model.config.vocab_size == target_model.config.vocab_size

    print("\n-- forward check (greedy spec-decode vs target-only greedy) --")
    fwd = forward_check(draft_model, target_model, tokenizer)

    print("\n-- reverse check (fault injection: verifier must flag planted bugs) --")
    rev = reverse_check(draft_model, target_model, tokenizer)

    n_match = sum(r["match_vs_target_only"] for r in fwd)
    forward_pass = n_match == len(fwd)
    reverse_pass = all(v["verifier_catches_this_bug"] for v in rev.values())

    result = {
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "gamma": GAMMA,
        "max_new_tokens": MAX_NEW_TOKENS,
        "forward_check": {
            "prompts_matching_target_only": n_match,
            "prompts_total": len(fwd),
            "pass": forward_pass,
            "per_prompt": fwd,
        },
        "reverse_check": {"pass": reverse_pass, "bugs": rev},
        "p1_2_pass": bool(forward_pass and reverse_pass),
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"forward check : {n_match}/{len(fwd)} prompts identical  -> {'PASS' if forward_pass else 'FAIL'}")
    print(f"reverse check : {'PASS' if reverse_pass else 'FAIL'}")
    print(f"P1.2 overall  : {'PASS' if result['p1_2_pass'] else 'FAIL'}")
    print(f"written to {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
