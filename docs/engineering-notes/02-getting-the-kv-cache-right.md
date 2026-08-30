# KV-cache rollback anchor: prefix + n_accepted, not prefix + k + 1

**Task:** P6.0 — single-sequence speculative decoding with a real KV cache.
**Pitfall:** 18.

## Why this step exists

The M2 speculative decoder was correct but cacheless: every round re-ran both
models over the whole prefix. On this Mac that's brutal — target-only decoding
with a KV cache runs at **24.4 tok/s**, without one at **9.0 tok/s** (a 2.7×
gap). A speculative decoder that throws the cache away every round is competing
with one hand tied behind its back. P6.0 is where the cache gets wired through
the accept/reject loop.

## The plan's formula was wrong

The design doc said: on a *partial* acceptance (draft proposed γ tokens, target
accepted `k < γ` of them), crop the target KV cache to `prefix + k + 1` and the
draft cache to `prefix + k` — two different lengths, with "the extra slot on the
target side for the bonus / resampled token."

That last clause is the mistake. Walk through one round:

- The target forward is fed `[pending_target_token] + [γ draft tokens]` — that's
  `γ + 1` input positions — and produces `γ + 1` rows of logits.
- On a rejection at position `k`, the token you emit at `k` is **resampled from**
  `(target_logits[k] − draft_logits[k])₊`. On a full acceptance, the bonus token
  is sampled from `target_logits[γ]`. Either way, that token is the **output** of
  the forward.
- It was never an **input** to any forward. No forward has ever attended to it.
  So there is no KV for it in either cache.

The correct rollback is: **crop both caches to `prefix + n_accepted`** (and
`prefix + γ` on a full acceptance). Same length on both sides. The resampled or
bonus token becomes the *first* input of the next round, and its KV gets created
then, like every other token's.

## Why the bug hides

On a **full-acceptance** round, `prefix + k + 1` with `k = γ` equals the current
sequence length. A crop helper written as "if the target length is already
`≥` where we want it, return" does nothing — correctly, by accident. If your
test prompts have a high acceptance rate (this pair sits at α ≈ 0.77), most
rounds are full acceptances, and the wrong formula passes.

You only see it on a round with a genuine rejection, where `prefix + k + 1`
overshoots the real cache contents by one and the next round's positions are
off by one from there on — which shows up as a slow quality drift, not a crash.

## The fix

- Implement `prefix + n_accepted` for both caches.
- Wrap cropping in `_crop_to(cache, target_len)` that **asserts
  `target_len ≤ current`** — so a formula that asks for a longer cache fails
  loudly instead of silently no-op'ing. This also pins down
  `DynamicCache.crop`'s negative-offset semantics in one place.
- Add an always-on invariant, checked after every round:
  `cache.get_seq_length() == len(committed_tokens) - 1`
  (the cache trails the committed sequence by exactly the one token that's about
  to be fed next).
- Add a **rejection-heavy stress phase** to the harness (`phase = 2.4`) with
  prompts and a temperature that force partial-acceptance rounds, so the path
  that exposes the bug is actually on the test menu.
- Cross-check against HuggingFace's reference:
  `generation/utils.py::_speculative_sampling` does
  `past_key_values.crop(-(candidate_length - n_matches))`, i.e.
  `new_len = current - (γ - k) = prefix + k`. Same anchor.

## The result

With the cache correct, `spec_kv` reproduces target-only greedy output
**token-exact** across γ ∈ {1, 3, 5}, 8 prompts, 3 seeds — prefix-match fraction
1.0, minimum 1.0. Throughput lands at **0.93–1.0× of KV-cached target-only** on
this 0.5B/1.5B pair on this Mac: speculative decoding is at parity here, not a
speedup, because the draft forward isn't cheap enough relative to the target on
Apple silicon at batch 1. The honest headline is "correct, cache-efficient, and
this model pair on this hardware is in the dead zone" — see
[note 05](05-fake-quant-vs-real-int4.md) and Pitfall 4.

## The lesson

When you roll back a cache, the length you roll back to is determined by **how
many tokens were fed through a forward**, not by how many tokens the round
logically produced or touched. Anything the round *emitted* as an output has no
KV yet. And write the crop guard as an assertion that the target length can only
shrink — a rollback helper that silently accepts "grow to N" will hide an
off-by-one for as long as your prompts keep accepting.
