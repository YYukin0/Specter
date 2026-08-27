"""
P1.0 前置存活性 gate — see notes/project_plan_v9.md §7 P1.0.

1. Assert draft/target tokenizers share an identical vocab (坑1: same-family
   models are not guaranteed to share a vocab).
2. Measure greedy-mode speculative-decoding acceptance rate alpha across a
   small prompt set (full probabilistic rejection sampling with correctness
   proof is P1.1, not this gate).
3. Apply the two-tier gate: alpha<0.4 fail, 0.4<=alpha<0.65 caution,
   alpha>=0.65 pass.

Model pair note: the plan tentatively assumed Qwen2.5-0.5B-Instruct (draft) /
Qwen2.5-7B-Instruct (target). This machine had ~9GB of free+inactive RAM at
the time of this run (out of 24GB unified memory, with several other apps
already resident) — loading both models in bf16 (0.5B ~1GB + 7B ~14GB =
~15GB of weights alone) risked swapping. Qwen2.5-3B-Instruct was substituted
as target to stay well within the observed headroom; this is a documented
swap, not an assumption — rerun with the 7B target if more headroom is
available.
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPTS

DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "mps"
DTYPE = torch.bfloat16
GAMMA = 4
MAX_NEW_TOKENS = 40
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p1_0_gate_result_b_track.json"


def check_vocab(draft_tok, target_tok):
    draft_vocab = draft_tok.get_vocab()
    target_vocab = target_tok.get_vocab()
    match = draft_vocab == target_vocab
    print(f"draft vocab_size={len(draft_vocab)}  target vocab_size={len(target_vocab)}")
    if not match:
        only_draft = list(set(draft_vocab) - set(target_vocab))
        only_target = list(set(target_vocab) - set(draft_vocab))
        print(f"  tokens only in draft ({len(only_draft)}): {only_draft[:5]}")
        print(f"  tokens only in target ({len(only_target)}): {only_target[:5]}")
    return match


@torch.no_grad()
def run_prompt(draft_model, target_model, tokenizer, eos_ids, prompt, gamma, max_new_tokens, device):
    messages = [{"role": "user", "content": prompt}]
    context = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"].to(device)

    total_proposed = 0
    total_accepted = 0
    generated = 0

    while generated < max_new_tokens:
        gamma_cur = min(gamma, max_new_tokens - generated)

        draft_out = draft_model.generate(
            context, max_new_tokens=gamma_cur, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        draft_tokens = draft_out[:, context.shape[1]:]
        gamma_cur = draft_tokens.shape[1]
        if gamma_cur == 0:
            break

        candidate = torch.cat([context, draft_tokens], dim=1)
        logits = target_model(candidate).logits
        start = context.shape[1] - 1
        target_argmax = logits[:, start:start + gamma_cur + 1, :].argmax(dim=-1)

        n_accept = 0
        for i in range(gamma_cur):
            total_proposed += 1
            if draft_tokens[0, i].item() == target_argmax[0, i].item():
                n_accept += 1
                total_accepted += 1
            else:
                break

        if n_accept == gamma_cur:
            new_tokens = draft_tokens[0, :n_accept].tolist() + [target_argmax[0, gamma_cur].item()]
        else:
            new_tokens = draft_tokens[0, :n_accept].tolist() + [target_argmax[0, n_accept].item()]

        context = torch.cat(
            [context, torch.tensor([new_tokens], device=device, dtype=context.dtype)], dim=1
        )
        generated += len(new_tokens)

        if new_tokens[-1] in eos_ids:
            break

    alpha = total_accepted / total_proposed if total_proposed else 0.0
    return {
        "prompt": prompt,
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "alpha": alpha,
        "generated_tokens": generated,
    }


def gate_decision(alpha):
    if alpha < 0.4:
        return "FAIL", "alpha < 0.4 -- this model pair does not work for speculative decoding, pick a different pair."
    elif alpha < 0.65:
        return "CAUTION", "0.4 <= alpha < 0.65 -- proceed to P1.1 but flag lowered expectations for downstream speedup."
    else:
        return "PASS", "alpha >= 0.65 -- proceed to P1.1 normally."


def main():
    print(f"draft  = {DRAFT_MODEL_ID}")
    print(f"target = {TARGET_MODEL_ID}")
    print(f"device = {DEVICE}, dtype = {DTYPE}, gamma = {GAMMA}")
    print()

    print("loading tokenizers...")
    draft_tok = AutoTokenizer.from_pretrained(DRAFT_MODEL_ID)
    target_tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

    vocab_match = check_vocab(draft_tok, target_tok)
    print(f"vocab match: {vocab_match}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)

    if not vocab_match:
        result = {
            "draft_model": DRAFT_MODEL_ID,
            "target_model": TARGET_MODEL_ID,
            "vocab_match": False,
            "gate_tier": "FAIL",
            "conclusion": "GATE FAILED at vocab check -- models do not share a tokenizer. Pick a different pair.",
        }
        RESULTS_PATH.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    print()

    print("loading models (this may take a while)...")
    t0 = time.time()
    draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    print(f"models loaded in {time.time() - t0:.1f}s")

    eos_ids = set()
    for tok_eos in (draft_tok.eos_token_id, target_tok.eos_token_id):
        if isinstance(tok_eos, list):
            eos_ids.update(tok_eos)
        elif tok_eos is not None:
            eos_ids.add(tok_eos)
    gen_eos = getattr(target_model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, list):
        eos_ids.update(gen_eos)
    elif gen_eos is not None:
        eos_ids.add(gen_eos)
    print(f"eos token ids: {eos_ids}")
    print()

    per_prompt_results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i + 1}/{len(PROMPTS)}] {prompt[:60]!r}")
        r = run_prompt(draft_model, target_model, target_tok, eos_ids, prompt, GAMMA, MAX_NEW_TOKENS, DEVICE)
        print(f"    accepted {r['total_accepted']}/{r['total_proposed']}  alpha={r['alpha']:.3f}")
        per_prompt_results.append(r)

    total_accepted = sum(r["total_accepted"] for r in per_prompt_results)
    total_proposed = sum(r["total_proposed"] for r in per_prompt_results)
    overall_alpha = total_accepted / total_proposed if total_proposed else 0.0

    tier, message = gate_decision(overall_alpha)

    result = {
        "draft_model": DRAFT_MODEL_ID,
        "target_model": TARGET_MODEL_ID,
        "gamma": GAMMA,
        "max_new_tokens_per_prompt": MAX_NEW_TOKENS,
        "vocab_match": True,
        "per_prompt": per_prompt_results,
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
        "overall_alpha": overall_alpha,
        "gate_tier": tier,
        "conclusion": message,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2))

    print()
    print("=" * 60)
    print(f"overall alpha (greedy-mode acceptance) = {overall_alpha:.4f}  ({total_accepted}/{total_proposed})")
    print(f"gate tier: {tier}")
    print(message)
    print("=" * 60)


if __name__ == "__main__":
    main()
