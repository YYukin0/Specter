# AWQ calibration-size ablation: the flat curve was a stuck knob, not a finding

**Task:** P2.3 — how much calibration data does AWQ actually need?
**Pitfall:** 16.

## The setup

The AWQ paper's selling point is that activation-aware scaling needs only a
*small* calibration set — a few dozen sequences — to pick per-channel scales,
where GPTQ-style methods are more data-hungry. I wanted to reproduce that on the
1.5B target: sweep `n_calib ∈ {4, 8, 16, 32, 64}`, quantize at each point, plot
wikitext-2 perplexity, and confirm the curve goes flat early.

I ran it. The curve wasn't just flat — it was **identical**: seed 0 gave
perplexity `13.6015` at `n_calib = 4`, and `13.6015` at 8, at 16, at 32, at 64,
bit-for-bit, same quantized weights, same eval.

That result matched the paper's claim and my own expectation going in ("AWQ is
sample-efficient enough that 4 sequences already saturate it"), and I nearly
wrote it up as a clean reproduction on that basis.

## What was actually happening

The self-built AWQ path captures the input activations of all 196 target Linear
layers in a single calibration forward, so it needs a per-layer cap to keep
memory bounded. That cap was hard-wired to 512 tokens, and each calibration
*sequence* was also truncated to 512 tokens.

So the first calibration sequence — 512 tokens — filled every layer's activation
pool to the cap, and capture stopped. Sequences 2 through 64 were never looked
at. Every point in my "sweep" fed the quantizer the byte-identical
"first 512 tokens of wikitext" tensor. The knob I was turning wasn't connected
to anything.

The flat curve was real. It just wasn't measuring calibration-set size.

## The tell I ignored

The review question I asked was *"is the curve flat?"* — and a flat curve was
what the hypothesis predicted, so I stopped. The question I should have asked
was *"did the independent variable actually move?"* Those are different
questions, and only the second one can fail in a way that catches this bug.

This is textbook confirmation bias, and it's worse in ML-systems work than
elsewhere because a broken knob and a genuinely-insensitive knob produce the
*same plot*. You cannot tell them apart by looking at the outcome. You have to
instrument the mechanism.

## The fix

- Expose `max_tokens_per_layer` and `max_seq_len` as real parameters. Default
  stays 512 so the earlier P2.2 run is byte-for-byte unchanged.
- In P2.3, truncate calibration rows to 64 tokens and set the cap to
  `n_calib × 64`, so the activation pool grows linearly with the knob.
- Record `captured_tokens_per_layer` at every sweep point.
- Emit a `capture_knob_actually_moved` boolean — false when
  `max(captured) / min(captured) ≤ 1.5×` — and have it auto-annotate the verdict
  string. If the knob didn't move, the result file *says so* instead of
  presenting a flat curve as a finding.
- Drop `n_calib = 128`: with the linear cap it would OOM a 24 GB machine.

## The lesson

When an ablation's result matches your hypothesis, that's exactly when to check
that the independent variable moved. A confirmed prediction is not evidence
until you've ruled out "the experiment did nothing." Add an assertion on the
*input* distribution, not just a plot of the output.
