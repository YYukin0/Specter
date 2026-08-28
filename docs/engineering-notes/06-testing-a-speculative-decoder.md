# Testing a speculative decoder: what output-equivalence checks miss

**Task:** P6.5 — a fault-injection / mutation-testing methodology for
speculative-decoding implementations.
**Pitfalls:** 坑2, 坑13, and the M-KV / M-POS blind spots below.

## The obvious test, and why it isn't enough

Speculative decoding has one headline correctness property: **the output
distribution is identical to running the target model alone.** At temperature 0
that's "same tokens"; with sampling it's "same tokens under a matched seed." So
the obvious test is: generate with the spec decoder, generate target-only,
`assert equal`.

That test is necessary and it is *weak*. It catches faults that flip an emitted
token. It is blind to any bug that corrupts internal state in a way that happens
not to change the argmax on the prompts you tried. And a lot of real
speculative-decoding bugs are exactly that shape — a KV cache cropped to the
wrong length ([note 02](02-getting-the-kv-cache-right.md)), a position index off
by one, a bonus token drawn from the wrong distribution (坑2). On a high-
acceptance model pair at greedy, several of those produce identical output for
dozens of tokens before drifting, or never drift on a short prompt at all.

## The oracle lattice

So P6.5 builds a stack of oracles, cheap-and-strict to expensive-and-loose, and
runs a library of **mutation operators** (`spec_faultlib` — 20+ deliberate bugs:
inverted accept ratios, frozen `cache_position`, `abs` instead of `relu` in the
residual distribution, EOS ignored mid-block, off-by-one positions, …) through
all of them.

| Oracle | Model | What it checks | Strength |
|--------|-------|----------------|----------|
| **O1** | `FakeModel` (symbolic, position-one-hot logits) | greedy output is *exactly* what the math predicts | strictest; kills the most operators |
| **O3** | `FakeModel` | sampling output under a matched seed | catches RNG-path / draw-order bugs |
| **O4** | any | structural invariants: cache length == committed − 1, positions monotone, `cache_position` vector correct, accept count in `[0, γ]` | **always on**, model-independent |
| **O5** | `FakeModel` | batch path == N × single-sequence, bit-exact | catches batch desync ([note 03](03-the-batch-correctness-tax.md)) |
| **O2** | **real Qwen2.5**, CPU, fp32 | greedy output exactly matches real target-only | most realistic; loosest |

The `FakeModel` matters: its logits are a one-hot function of position, so it has
no RoPE smoothing and *no tolerance* for a position error. That makes O1
brutal — it kills all three M-POS (`cache_position` shift/freeze) operators
outright.

## The finding: a real model is not a superset of the fake one

I built O2 (real model, CPU/fp32, greedy-exact) expecting it to dominate O1 —
"if the fake model catches it, surely the real one does too." It doesn't.

O2 results across 3 prompts, γ=4, 40 tokens, 9 mutation operators that O1 kills:

- **Caught (5):** `adjusted_abs_not_relu`, `accept_ratio_inverted`,
  `accept_always`, `bonus_token_from_draft` (only after 33 tokens —
  provenance bugs need a long enough generation), `force_accept_first`.
- **Missed (4):** all three M-POS operators (`pos_id_frozen`,
  `pos_id_off_by_one_plus`, `pos_id_off_by_one_minus`), plus
  `eos_ignored_midblock` (no test prompt actually hit EOS mid-block).

A real RoPE model at greedy **shrugs off a small or collapsed position error**
on these prompts — the logit gap between the top-1 and top-2 token is usually
wide enough to absorb it. The position-one-hot `FakeModel` cannot; every
position error moves its argmax. So:

> **A real-model output oracle is not a superset of the FakeModel output
> oracle.** The FakeModel *over-represents* position sensitivity, and that's a
> feature — it's why O1 catches M-POS. Neither output oracle, fake or real, is a
> substitute for the structural check.

M-POS faults are caught by **O4** (which reads the `cache_position` vector
directly and checks it against the expected ramp) and by **specdiff**'s
`UPSTREAM_KV_POS` rule — not by any "compare the tokens" test at any model
fidelity.

## Two more sub-findings

- **Batch invariance is bitwise on the reference path.** CPU/fp32 single-forward
  vs row-0-of-a-left-padded-batch last-token logits are **identical, max delta
  0.0**, given a correct attention mask and `position_ids` derived from the mask.
  arXiv:2607.17283's reported 5.8e-3 per-logit batch delta is a quantized-Metal
  backend artifact, not an algorithmic property. So when specdiff sees "every
  structural signal agrees, only the committed prefix differs," it routes to
  `BACKEND_NONDETERMINISM`, not to a math or cache mechanism — and there's a
  hermetic test pinning that classification contract.

- **Budget overshoot is not a divergence.** `speculative_generate_kv` emits a
  whole round at once, so it routinely overshoots `max_new_tokens` by up to γ
  (often landing on EOS). A trailing length gap `≤ γ` with an identical shared
  prefix is *correct behaviour*; the oracle's equality check has to special-case
  it or it reports false divergences on clean code.

## The lesson

"The spec decoder's output matches target-only" is the property you ship, but
it's the *weakest* test you can write for it. Build a lattice: a symbolic model
strict enough to have no error tolerance, structural invariants that don't depend
on any model, a batch-equivalence check, and a real-model check — and expect each
one to catch a different class of bug. In particular, don't assume the
most-realistic oracle is the most-powerful one. The fake model catches the
position bugs precisely *because* it's unrealistically brittle.
