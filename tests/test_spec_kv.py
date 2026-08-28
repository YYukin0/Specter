"""Tests for KV-cache-correct speculative decoding (src/spec_kv.py).

Contract (deployment-depth plan P6.0):
  (1) FakeModel greedy: speculative_generate_kv == speculative_generate ==
      target_only_generate, token for token.
  (2) FakeModel sampling, shared seed: speculative_generate_kv ==
      speculative_generate, token for token (identical generator-draw order).
  (3) after every round both caches hold exactly len(committed) - 1 positions
      (the trailing resample/bonus token is always next round's pending input --
      this is the HF assisted-generation invariant; the plan's "== committed"
      wording is off by that one token).
  (4) rollback stress: a ~0.5-acceptance FakeModel drives many partial rollbacks
      and parity + the cache-length invariant still hold.
  (5) 坑13: an EOS mid-chunk truncates at the same place as the no-cache path.

Real MPS fp16 parity (long-common-prefix only) is checked in src/verify_spec_kv.py,
not here -- the pytest suite stays hermetic and CPU-only.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rejection_sampling import Injection, speculative_generate, target_only_generate  # noqa: E402
from spec_kv import (  # noqa: E402
    speculative_generate_kv,
    speculative_step_kv,
    target_only_generate_kv,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeCache:
    """Length-only stand-in for DynamicCache: enough for the cache-length
    invariant assertions, none of the real key/value storage."""

    def __init__(self):
        self._len = 0

    def _grow(self, n):
        self._len += n

    def get_seq_length(self):
        return self._len

    def crop(self, tokens_to_remove):
        # spec_kv always drives crop with a negative argument
        assert tokens_to_remove < 0
        self._len += tokens_to_remove
        assert self._len >= 0


class KVFake:
    """Position-only logits: row for absolute position p depends on p alone, not
    on token identity or on whether a KV cache was used -- so a cached forward is
    bit-identical to a full re-forward. `phase` shifts draft vs target apart."""

    def __init__(self, vocab=16, phase=0.0, eos_id=None, eos_after=None):
        self.vocab = vocab
        self.phase = phase
        self.eos_id = eos_id
        self.eos_after = eos_after
        self.generation_config = type("g", (), {"eos_token_id": eos_id})()

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=None, cache_position=None):
        B, L = input_ids.shape
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past, past + L)
        v = torch.arange(self.vocab).float()
        logits = torch.zeros(B, L, self.vocab)
        for b in range(B):
            for i in range(L):
                p = float(cache_position[i])
                logits[b, i] = torch.sin(v * 0.9 + p * 0.4 + self.phase) * 2.0 \
                    + torch.cos(v * 0.3 - p * 0.2)
                if self.eos_id is not None and self.eos_after is not None and p >= self.eos_after:
                    logits[b, i, self.eos_id] = 20.0
        if past_key_values is not None and hasattr(past_key_values, "_grow"):
            past_key_values._grow(L)
        return _Out(logits)


class _Tok:
    eos_token_id = None
    pad_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt"):
        content = msgs[0]["content"]
        n = 3 + (len(content) % 4)
        return torch.arange(1, 1 + n).unsqueeze(0)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


PROMPTS = ["hello world", "speculative", "a longer prompt here", "kv"]


# --------------------------------------------------------------------------- #
# (1) greedy parity: kv == no-cache spec == target-only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gamma", [1, 3, 5])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_parity_all_three(prompt, gamma):
    draft = KVFake(phase=0.0)
    target = KVFake(phase=0.6)
    tok = _Tok()
    kw = dict(gamma=gamma, max_new_tokens=32, temperature=0.0, seed=0)

    kv = speculative_generate_kv(prompt, draft, target, tok, make_cache=FakeCache, **kw)
    nc = speculative_generate(prompt, draft, target, tok, **kw)
    to = target_only_generate(prompt, target, tok, max_new_tokens=32, temperature=0.0, seed=0)

    # exact KV-vs-no-cache parity (the real contract: a cache changes nothing)
    assert kv.token_ids == nc.token_ids
    # vs target-only: identical up to the shared length. speculative_generate can
    # overshoot max_new_tokens by up to gamma on its last round (g is clamped but
    # a round still emits k+1) -- that overshoot is inherited, not a KV bug.
    n = min(len(kv.token_ids), len(to.token_ids))
    assert kv.token_ids[:n] == to.token_ids[:n]
    assert abs(len(kv.token_ids) - len(to.token_ids)) <= gamma


# --------------------------------------------------------------------------- #
# (2) sampling parity: kv == no-cache spec, shared seed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("gamma", [1, 3, 4])
def test_sampling_parity_shared_seed(gamma, seed):
    draft = KVFake(phase=0.0)
    target = KVFake(phase=0.7)
    tok = _Tok()
    kw = dict(gamma=gamma, max_new_tokens=40, temperature=1.0, seed=seed)
    for prompt in PROMPTS:
        kv = speculative_generate_kv(prompt, draft, target, tok, make_cache=FakeCache, **kw)
        nc = speculative_generate(prompt, draft, target, tok, **kw)
        assert kv.token_ids == nc.token_ids, f"prompt={prompt!r} seed={seed} gamma={gamma}"
        assert kv.accept_lengths == nc.accept_lengths
        assert kv.alpha == pytest.approx(nc.alpha)


# --------------------------------------------------------------------------- #
# (2b) record mode keeps parity (the extra draft bonus-row forward must not
#      perturb the generator-draw order)
# --------------------------------------------------------------------------- #
def test_record_mode_parity():
    draft = KVFake(phase=0.0)
    target = KVFake(phase=0.7)
    tok = _Tok()
    kw = dict(gamma=4, max_new_tokens=40, temperature=1.0, seed=1, record=True)
    kv = speculative_generate_kv("hello world", draft, target, tok, make_cache=FakeCache, **kw)
    nc = speculative_generate("hello world", draft, target, tok, **kw)
    assert kv.token_ids == nc.token_ids
    assert len(kv.proposals) == len(nc.proposals)


# --------------------------------------------------------------------------- #
# (3) cache-length invariant: both caches == len(committed) - 1 after each round
# --------------------------------------------------------------------------- #
def _drive_with_invariant(prompt, draft, target, tok, gamma, temperature, seed, phase_note=""):
    """Mirror speculative_generate_kv's loop but assert the cache-length
    invariant after every round. Returns the emitted token ids."""
    from rejection_sampling import collect_eos_ids, encode_prompt

    gen = torch.Generator()
    gen.manual_seed(seed)
    device = torch.device("cpu")
    ctx = encode_prompt(tok, prompt, device, True)
    committed = ctx[0].tolist()
    eos_ids = collect_eos_ids(tok, target)
    dcache, tcache = FakeCache(), FakeCache()
    dsync = tsync = 0
    out = []
    max_new = 48
    while len(out) < max_new:
        g = min(gamma, max_new - len(out))
        step = speculative_step_kv(
            committed, draft, target, dcache, tcache, dsync, tsync, g,
            device=device, dtype=ctx.dtype, temperature=temperature, generator=gen,
        )
        dsync, tsync = step.draft_synced, step.target_synced
        emitted = step.result.new_token_ids
        hit = False
        for k, tid in enumerate(emitted):
            if tid in eos_ids:
                emitted = emitted[: k + 1]
                hit = True
                break
        out.extend(emitted)
        committed.extend(emitted)
        if not hit:
            # invariant holds on every non-terminal round
            assert dcache.get_seq_length() == len(committed) - 1, phase_note
            assert tcache.get_seq_length() == len(committed) - 1, phase_note
        assert dsync == dcache.get_seq_length()
        assert tsync == tcache.get_seq_length()
        if hit:
            break
    return out


@pytest.mark.parametrize("gamma", [1, 2, 3, 5])
def test_cache_length_invariant(gamma):
    draft = KVFake(phase=0.0)
    target = KVFake(phase=0.6)
    tok = _Tok()
    got = _drive_with_invariant("hello world", draft, target, tok, gamma, 0.0, 0)
    ref = speculative_generate("hello world", draft, target, tok, gamma=gamma,
                               max_new_tokens=48, temperature=0.0, seed=0)
    assert got == ref.token_ids


# --------------------------------------------------------------------------- #
# (4) rollback stress: ~0.5 acceptance -> many partial rollbacks
# --------------------------------------------------------------------------- #
def test_rollback_stress_low_acceptance():
    # large phase gap -> draft and target disagree often -> lots of k < gamma
    draft = KVFake(phase=0.0)
    target = KVFake(phase=2.4)
    tok = _Tok()
    for seed in (0, 1, 2, 3, 4):
        kw = dict(gamma=5, max_new_tokens=60, temperature=1.0, seed=seed)
        kv = speculative_generate_kv("a longer prompt here", draft, target, tok,
                                     make_cache=FakeCache, **kw)
        nc = speculative_generate("a longer prompt here", draft, target, tok, **kw)
        assert kv.token_ids == nc.token_ids, f"seed {seed}"
        # confirm the stress actually happened: some rounds partially rejected
        assert any(a < 5 for a in kv.accept_lengths)
    # and the cache-length invariant survives the same stress
    _drive_with_invariant("a longer prompt here", draft, target, tok, 5, 1.0, 2,
                          phase_note="rollback stress")


# --------------------------------------------------------------------------- #
# (5) 坑13: EOS mid-chunk truncates identically
# --------------------------------------------------------------------------- #
def test_eos_midblock_parity():
    eos = 7
    draft = KVFake(phase=0.0)
    target = KVFake(phase=0.6, eos_id=eos, eos_after=8)

    class _TokE(_Tok):
        eos_token_id = eos

    tok = _TokE()
    for seed in (0, 1, 2):
        kw = dict(gamma=4, max_new_tokens=40, temperature=1.0, seed=seed)
        kv = speculative_generate_kv("kv", draft, target, tok, make_cache=FakeCache, **kw)
        nc = speculative_generate("kv", draft, target, tok, **kw)
        assert kv.token_ids == nc.token_ids, f"seed {seed}"
        assert kv.token_ids[-1] == eos
        assert kv.token_ids.count(eos) == 1


# --------------------------------------------------------------------------- #
# (6) fault injection still visible through the KV path (non-blindness)
# --------------------------------------------------------------------------- #
def test_injection_bonus_from_draft_diverges():
    draft = KVFake(phase=0.0)
    target = KVFake(phase=0.7)
    tok = _Tok()
    kw = dict(gamma=3, max_new_tokens=40, temperature=1.0, seed=0)
    clean = speculative_generate_kv("hello world", draft, target, tok, make_cache=FakeCache, **kw)
    bugged = speculative_generate_kv("hello world", draft, target, tok, make_cache=FakeCache,
                                     injection=Injection(bonus_from_draft=True), **kw)
    # only bites on full-accept rounds; with this phase gap there is at least one
    assert clean.token_ids != bugged.token_ids


# --------------------------------------------------------------------------- #
# target_only_generate_kv parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_target_only_kv_parity(temperature):
    target = KVFake(phase=0.6)
    tok = _Tok()
    for prompt in PROMPTS:
        kv = target_only_generate_kv(prompt, target, tok, make_cache=FakeCache,
                                     max_new_tokens=32, temperature=temperature, seed=0)
        nc = target_only_generate(prompt, target, tok, max_new_tokens=32,
                                  temperature=temperature, seed=0)
        assert kv.token_ids == nc.token_ids


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
