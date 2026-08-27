"""Tests for batched speculative decoding (src/spec_batch.py).

Anchors:
  (a) batch_size=1 reproduces speculative_generate token-for-token at one seed
      (the acceptance math + generator-draw order are correct).
  (b) batch_size>1 runs to completion with ragged per-seq accept lengths, and
      EOS is handled independently per sequence.
  (c) left-padding + position_ids + attention_mask do not shift an unpadded
      sequence's logits (white-box check on the target-phase row extraction).

Note: a single shared torch.Generator is consumed in sequence order, so the
SPECIFIC tokens a given sequence gets depend on batch composition (still a valid
draw from the correct joint distribution). Parity is asserted only for
batch_size=1, which is the documented contract.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rejection_sampling import speculative_generate  # noqa: E402
from spec_batch import _fwd, _pad_batch, speculative_generate_batch  # noqa: E402


class _Out:
    def __init__(self, logits):
        self.logits = logits


class BatchFake:
    """Position-only logits (depend on position_ids, not token identity) so a
    left-padded sequence with correct position_ids is bit-identical to the same
    sequence run alone. `phase` shifts the distribution to make draft != target."""

    def __init__(self, vocab=16, phase=0.0, eos_id=None, eos_after=None):
        self.vocab = vocab
        self.phase = phase
        self.eos_id = eos_id
        self.eos_after = eos_after
        self.generation_config = type("g", (), {"eos_token_id": eos_id})()

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None):
        B, L = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(L).unsqueeze(0).expand(B, L)
        v = torch.arange(self.vocab).float()
        logits = torch.zeros(B, L, self.vocab)
        for b in range(B):
            for p in range(L):
                pp = float(position_ids[b, p])
                logits[b, p] = torch.sin(v * 0.9 + pp * 0.4 + self.phase) * 2.0 \
                    + torch.cos(v * 0.3 - pp * 0.2)
                if self.eos_id is not None and self.eos_after is not None and pp >= self.eos_after:
                    logits[b, p, self.eos_id] = 20.0
        return _Out(logits)


class _Tok:
    eos_token_id = None
    pad_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt"):
        content = msgs[0]["content"]
        n = 2 + (len(content) % 3)
        return torch.arange(1, 1 + n).unsqueeze(0)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def test_batch_size_1_matches_speculative_generate():
    draft = BatchFake(phase=0.0)
    target = BatchFake(phase=0.6)
    tok = _Tok()
    for seed in (0, 1, 2):
        ref = speculative_generate("hello world", draft, target, tok, gamma=3,
                                   max_new_tokens=20, temperature=1.0, seed=seed)
        got = speculative_generate_batch(["hello world"], draft, target, tok, gamma=3,
                                         max_new_tokens=20, temperature=1.0, seed=seed,
                                         batch_size=1)
        assert got.token_ids[0] == ref.token_ids, f"seed {seed}"
        assert got.mean_accept_length == pytest.approx(
            sum(ref.accept_lengths) / len(ref.accept_lengths))


def test_batch_runs_to_completion_with_ragged_accepts():
    draft = BatchFake(phase=0.0)
    target = BatchFake(phase=0.6)
    tok = _Tok()
    prompts = ["a", "bb", "ccc", "dddd"]
    r = speculative_generate_batch(prompts, draft, target, tok, gamma=3,
                                   max_new_tokens=24, temperature=1.0, seed=0,
                                   batch_size=4)
    assert len(r.token_ids) == 4
    assert all(len(t) >= 24 for t in r.token_ids)          # no EOS -> all hit the cap
    accept_sets = [tuple(s["accept_lengths"]) for s in r.per_seq]
    assert len(set(accept_sets)) > 1                        # genuinely ragged
    assert r.n_target_forwards >= 8                         # ~24/ (accept+1) rounds
    assert 0.0 < r.alpha <= 1.0


def test_eos_handled_independently_per_sequence():
    # target spikes EOS once a sequence is far enough along; different prompt
    # lengths -> different absolute positions -> different finish rounds
    eos = 7
    draft = BatchFake(phase=0.0)
    target = BatchFake(phase=0.6, eos_id=eos, eos_after=9)

    class _TokE(_Tok):
        eos_token_id = eos

    r = speculative_generate_batch(["a", "cccccc"], draft, target, _TokE(), gamma=3,
                                   max_new_tokens=40, temperature=1.0, seed=0,
                                   batch_size=2)
    for t in r.token_ids:
        assert eos in t
        assert t.index(eos) == len(t) - 1                   # truncated at first EOS
        assert len(t) < 40                                  # stopped early
    assert r.n_seq_finished_early == 2


def test_padding_does_not_shift_unpadded_logits():
    target = BatchFake(phase=0.6)
    short = [3, 4, 5]
    long = [1, 2, 3, 4, 5, 6, 7, 8]
    inp, attn, pos = _pad_batch([short, long], pad_id=0, device=torch.device("cpu"),
                                dtype=torch.long)
    batched = _fwd(target, inp, attn, pos)
    pad_off = inp.shape[1] - len(short)
    rows_from_batch = batched[0, pad_off:, :]

    s_inp, s_attn, s_pos = _pad_batch([short], pad_id=0, device=torch.device("cpu"),
                                      dtype=torch.long)
    solo = _fwd(target, s_inp, s_attn, s_pos)[0]
    assert torch.allclose(rows_from_batch, solo, atol=1e-5)


def test_tokens_per_target_forward_grows_with_batch():
    draft = BatchFake(phase=0.0)
    target = BatchFake(phase=0.6)
    tok = _Tok()
    prompts = ["a", "bb", "ccc", "dddd"]
    r1 = speculative_generate_batch(prompts, draft, target, tok, gamma=3,
                                    max_new_tokens=16, temperature=1.0, seed=0, batch_size=1)
    r4 = speculative_generate_batch(prompts, draft, target, tok, gamma=3,
                                    max_new_tokens=16, temperature=1.0, seed=0, batch_size=4)
    # same total work emitted, but batch=4 packs it into far fewer target calls
    assert r4.tokens_per_target_forward > r1.tokens_per_target_forward
    # per-seq-per-round efficiency should be similar (no KV cache, no math change)
    assert r4.mean_tokens_per_seq_per_round == pytest.approx(
        r1.mean_tokens_per_seq_per_round, rel=0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
