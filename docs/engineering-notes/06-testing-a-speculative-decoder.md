# Testing a speculative decoder: what output-equivalence checks miss

**Task:** P6.5 — a fault-injection / mutation-testing methodology for
speculative-decoding implementations.
**Pitfalls:** 坑2, 坑13, and the M-KV / M-POS blind spots below.

## Where this sits

Two separate correctness questions get conflated in inference work:

1. **Numerical reproducibility** — does the same input produce the same bits
   regardless of batch size, tensor-parallel degree, or kernel scheduling? This
   is the **batch-invariance** line of work: Thinking Machines' *Defeating
   Nondeterminism in LLM Inference* (He et al., 2025) and its batch-invariant
   RMSNorm / matmul / attention kernels, plus follow-ups on cross-TP determinism
   and verified speculation. That work lives at the **kernel** layer.
2. **Algorithmic control-flow correctness** — given deterministic kernels, does
   the *decoder loop* do the right thing? Did the KV cache get cropped to the
   right anchor on a partial acceptance? Is the position ramp contiguous? Is the
   bonus token drawn from the target's distribution and not the draft's?

This note is about (2). The bugs here are not floating-point noise; they are
off-by-one cache lengths and swapped distributions. The relevant prior art is
**mutation testing**, not determinism research — but the standard mutation-testing
tools don't fit, which is the first thing to explain.

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

In mutation-testing terms: those bugs are **mutants that survive the
output-equivalence test suite**. The question P6.5 answers is *which oracle you
have to add to kill them* — its **mutation adequacy**.

## Why not cosmic-ray / mutmut

Off-the-shelf Python mutation testing (cosmic-ray, mutmut, DeepCrime) mutates
**syntax**: flip a comparison operator, bump an integer literal, swap `and`/`or`,
delete a statement. That is a good fit for business logic. It is a poor fit here
for two reasons:

- **The interesting faults are semantic, not syntactic.** "Roll the KV cache
  back to `prefix + k + 1` instead of `prefix + n_accepted`" is not a
  one-token edit — it's a wrong *invariant*. "Draw the bonus token from the
  draft distribution" is a change of which tensor flows where. A literal-mutation
  engine can stumble onto some of these (`NumberReplacer` on a crop offset gives
  you the off-by-one) but can't express most of them.
- **The equivalent-mutant problem is severe in numerical code.** cosmic-ray
  bumping a constant that doesn't change any observable output produces an
  *equivalent mutant* — counted as "survived", inflating the noise floor,
  needing manual triage. Half of `spec_faultlib`'s operators would land there.

So P6.5 hand-writes **~20 semantic mutation operators** (`spec_faultlib`) grouped
by the part of the algorithm they corrupt — M-KV (cache management), M-POS
(position indices), M-SAMPLE (the acceptance / resampling math), M-CTRL (the
accept/reject control flow) — each a deliberate, named, toggleable bug. The
harness is custom; the *methodology* (define operators, run them past a battery
of oracles, report a kill matrix) is standard mutation testing.

## The oracle lattice

P6.5 runs every operator past a stack of oracles, cheap-and-strict to
expensive-and-loose:

| Oracle | Model | What it checks | Strength |
|--------|-------|----------------|----------|
| **O1** | `FakeModel` (symbolic, position-one-hot logits) | greedy output is *exactly* what the math predicts | strictest; kills the most operators |
| **O3** | `FakeModel` | sampling output under a matched seed | catches RNG-path / draw-order bugs |
| **O4** | any | structural invariants: cache length == committed − 1, positions monotone, `cache_position` vector correct, accept count in `[0, γ]` | **always on**, model-independent |
| **O5** | `FakeModel` | batch path == N × single-sequence, bit-exact | **a batch-invariance check** (see below) |
| **O2** | **real Qwen2.5**, CPU, fp32 | greedy output exactly matches real target-only | most realistic; loosest |

The `FakeModel` matters: its logits are a one-hot function of position, so it has
no RoPE smoothing and *no tolerance* for a position error. That makes O1
brutal — it kills all three M-POS (`cache_position` shift/freeze) operators
outright.

**O5 is not a new idea and this note doesn't claim it as one.** It is a
batch-invariance assertion — "does widening the batch change any sequence's
output" — which is exactly the property the determinism literature above
formalizes and builds kernels for. Specter's version is much narrower (one
reference path, a `FakeModel`, no custom kernels) and on that path it holds
**bitwise**, so O5 kills nothing on its own. It earns its place by being the
regression guard for the "output equivalence by construction" claim in
[note 03](03-the-batch-correctness-tax.md): the per-sequence-cache design has no
shared ragged tensor to desync, and O5 pins that under adversarial mutation.

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

## The mutation-adequacy matrix

`results/p6_5_mutation_adequacy.json`, ≥3 seeds. Kill rate per oracle over the
full operator set:

- **any oracle:** 1.0 — every mutant is caught by *something*.
- **O1** (greedy exact): 0.5625
- **O3** (sampling distribution): 0.75
- **O4** (structural invariants): 0.25 — low count, but it is the *only* oracle
  that kills the M-KV and M-POS operators.
- **O5** (batch equivalence): 0.0 on the reference path, as expected.

The headline: **every M-KV cache-management mutant is invisible to both the
greedy and the sampling output-equivalence oracle. Only O4's structural
assertions kill them.** `adjusted_no_renormalize` (drop the renormalization in
the residual distribution) is invisible to *every* output oracle because
`torch.multinomial` silently renormalizes — only O4's explicit sum-to-one
assertion catches it. In mutation-testing language: the output-equivalence suite
has a **mutation score of ~0.56–0.75 depending on sampling mode, and 0.0 against
the M-KV class**; you cannot buy that class back by making the model more
realistic, only by asserting on structure.

## Two more sub-findings

- **Batch invariance is bitwise on the reference path.** CPU/fp32 single-forward
  vs row-0-of-a-left-padded-batch last-token logits are **identical, max delta
  0.0**, given a correct attention mask and `position_ids` derived from the mask.
  arXiv:2607.17283's reported 5.8e-3 per-logit batch delta is a quantized-Metal
  backend artifact — a *kernel*-layer nondeterminism of exactly the kind the
  Thinking Machines work addresses — not an algorithmic property of the decoder.
  So when specdiff sees "every structural signal agrees, only the committed
  prefix differs," it routes to `BACKEND_NONDETERMINISM`, not to a math or cache
  mechanism — and there's a hermetic test pinning that classification contract.

- **Budget overshoot is not a divergence.** `speculative_generate_kv` emits a
  whole round at once, so it routinely overshoots `max_new_tokens` by up to γ
  (often landing on EOS). A trailing length gap `≤ γ` with an identical shared
  prefix is *correct behaviour*; the oracle's equality check has to special-case
  it or it reports false divergences on clean code.

## The lesson

"The spec decoder's output matches target-only" is the property you ship, but
it's the *weakest* test you can write for it — a mutation score well under 1.0,
and 0.0 against the whole cache-management fault class. Build a lattice: a
symbolic model strict enough to have no error tolerance, structural invariants
that don't depend on any model, a batch-invariance check, and a real-model check
— and expect each one to catch a different class of bug. In particular, don't
assume the most-realistic oracle is the most-powerful one. The fake model catches
the position bugs precisely *because* it's unrealistically brittle.
