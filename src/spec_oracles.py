"""
P6.5 Part 1 -- the oracle stack. "Correct" is a lattice, not one assertion.

Floating-point non-determinism makes a naive exact-match oracle both false-positive
and false-negative. The stack layers oracles by how much they can prove:

  O1  FakeModel, symbolic exact-match. A deterministic position-one-hot model, no
      FP noise. speculative_generate_kv(temperature=0) MUST equal a *clean*
      target_only reference token for token. The only exact oracle -- any
      divergence here is a real bug.
  O3  FakeModel, sampling mode (temperature=1.0), matched-seed exact-match
      against a *clean* speculative_generate_kv reference. On the deterministic
      FakeModel a mutation to the rejection-sampling *math* (accept ratio,
      adjusted distribution, bonus provenance, injected leniency) perturbs the
      generator-draw sequence and shows up here even when it is a no-op in greedy
      mode (where every distribution is one-hot). On a real model this same
      oracle becomes the plan's two-sample test with a vanilla-vs-vanilla null
      band; here it is a strict superset of O1.
  O4  structural invariants, always-on. Not output equivalence: the internal
      invariants every speculative step must satisfy regardless of the tokens
      produced -- cache lengths, accepted-count bounds, a valid adjusted
      distribution, a valid acceptance probability, seed determinism, no commit
      after EOS.

O2 (real model, CPU fp32, greedy exact) now lives in src/verify_p6_5_o2.py ->
results/p6_5_o2_real_model.json (needs model loads, not on the hermetic path).
Findings: clean greedy speculative == greedy target-only bit-exactly on the real
CPU/fp32 pair; O2 catches the accept-logic / adjusted-distribution / bonus faults
but MISSES all three M-POS (cache_position) faults that O1 kills on the
FakeModel -- a real RoPE model at greedy shrugs off a position shift the
position-one-hot FakeModel cannot, so a real-model output oracle is NOT a
superset of the fake one, and M-POS still needs O4 + specdiff's UPSTREAM_KV_POS
rule. The real-model / null-band form of O3 is still left for a later increment.

Used by src/verify_spec_faultlib.py to score every operator in
src/spec_faultlib.py -> results/p6_5_mutation_adequacy.json.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import rejection_sampling as _rs
import spec_faultlib as faultlib
import spec_kv as _kv
from rejection_sampling import encode_prompt
from spec_kv import speculative_generate_kv, speculative_step_kv, target_only_generate_kv
from spec_kv_batch import make_seq, run_round

# NB: call `_rs.collect_eos_ids(...)` through the module, never a `from ... import`
# binding -- the `eos_ignored_midblock` operator monkeypatches it on `_rs`, and a
# local binding would silently bypass the mutation (same trap as specdiff).


# --------------------------------------------------------------------------- #
# Deterministic fakes (oracles legitimately need a noise-free model)
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class PositionFakeModel:
    """Logits for absolute position p depend on p alone -- a cached forward is
    bit-identical to a full re-forward. `phase` separates draft from target;
    an optional EOS spike lets O1 see EOS-handling mutants."""

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


class LengthOnlyCache:
    """Tracks only sequence length. `crop` mirrors DynamicLayer.crop semantics
    (negative removes N; positive is the deprecated absolute-length form) so the
    kv_crop_absolute_vs_relative operator exercises the real branch."""

    def __init__(self):
        self._len = 0

    def _grow(self, n):
        self._len += n

    def get_seq_length(self):
        return self._len

    def crop(self, tokens_to_remove):
        if tokens_to_remove > 0:
            if tokens_to_remove < self._len:
                self._len = tokens_to_remove
            return
        self._len = max(0, self._len + tokens_to_remove)


class _Tok:
    eos_token_id = None
    pad_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt"):
        n = 3 + (len(msgs[0]["content"]) % 4)
        return torch.arange(1, 1 + n).unsqueeze(0)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class _TokEOS(_Tok):
    eos_token_id = 7


_PROMPTS = ["hello world", "speculative decode", "a longer prompt here",
            "kv", "the quick brown fox", "one two three four"]


def make_fake_pair(phase_target=0.6, eos=False):
    tok = _TokEOS() if eos else _Tok()
    draft = PositionFakeModel(phase=0.0, eos_id=7 if eos else None, eos_after=8 if eos else None)
    target = PositionFakeModel(phase=phase_target, eos_id=7 if eos else None, eos_after=8 if eos else None)
    return draft, target, tok


# --------------------------------------------------------------------------- #
# O1 -- FakeModel symbolic exact-match against a CLEAN reference
# --------------------------------------------------------------------------- #
@dataclass
class OracleResult:
    oracle: str
    killed: bool
    n_runs: int
    n_diverged: int
    first_divergence_tokens: List[int] = field(default_factory=list)  # per diverged run
    violations: List[str] = field(default_factory=list)
    note: str = ""


def _first_divergence(a: List[int], b: List[int], tol_tail: int) -> Optional[int]:
    """Index of the first mismatch within the shared length. A trailing length
    gap up to `tol_tail` (speculative decoding's documented max_new_tokens
    overshoot) is not itself a divergence."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if abs(len(a) - len(b)) > tol_tail:
        return n
    return None


def run_o1(mutants=(), *, gammas=(1, 3, 5), seeds=(0, 1, 2), eos=False,
           max_new_tokens=48, params=None) -> OracleResult:
    params = params or {}
    draft, target, tok = make_fake_pair(eos=eos)
    n_runs = n_div = 0
    firsts: List[int] = []
    for gamma in gammas:
        for seed in seeds:
            for prompt in _PROMPTS:
                # clean, unmutated reference
                ref = target_only_generate_kv(prompt, target, tok, make_cache=LengthOnlyCache,
                                              max_new_tokens=max_new_tokens, temperature=0.0,
                                              seed=seed)
                with faultlib.apply(*mutants, **params):
                    spec = speculative_generate_kv(prompt, draft, target, tok,
                                                   make_cache=LengthOnlyCache, gamma=gamma,
                                                   max_new_tokens=max_new_tokens,
                                                   temperature=0.0, seed=seed)
                n_runs += 1
                d = _first_divergence(spec.token_ids, ref.token_ids, tol_tail=gamma)
                if d is not None:
                    n_div += 1
                    firsts.append(d)
    return OracleResult(oracle="O1", killed=n_div > 0, n_runs=n_runs, n_diverged=n_div,
                        first_divergence_tokens=firsts)


# --------------------------------------------------------------------------- #
# O3 -- FakeModel sampling mode, matched-seed exact vs a clean spec reference
# --------------------------------------------------------------------------- #
def run_o3(mutants=(), *, gammas=(1, 3, 5), seeds=(0, 1, 2, 3, 4), eos=False,
          max_new_tokens=48, params=None) -> OracleResult:
    params = params or {}
    draft, target, tok = make_fake_pair(eos=eos)
    n_runs = n_div = 0
    firsts: List[int] = []
    for gamma in gammas:
        for seed in seeds:
            for prompt in _PROMPTS:
                ref = speculative_generate_kv(prompt, draft, target, tok,
                                              make_cache=LengthOnlyCache, gamma=gamma,
                                              max_new_tokens=max_new_tokens,
                                              temperature=1.0, seed=seed)
                with faultlib.apply(*mutants, **params):
                    got = speculative_generate_kv(prompt, draft, target, tok,
                                                  make_cache=LengthOnlyCache, gamma=gamma,
                                                  max_new_tokens=max_new_tokens,
                                                  temperature=1.0, seed=seed)
                n_runs += 1
                d = _first_divergence(got.token_ids, ref.token_ids, tol_tail=gamma)
                if d is not None:
                    n_div += 1
                    firsts.append(d)
    return OracleResult(oracle="O3", killed=n_div > 0, n_runs=n_runs, n_diverged=n_div,
                        first_divergence_tokens=firsts)


# --------------------------------------------------------------------------- #
# O5 -- batched path (P6.1) equals N independent single-sequence runs
# --------------------------------------------------------------------------- #
def _drive_batch(seqs, draft, target, tok, *, gamma, temperature, max_rounds=10**6):
    eos = _rs.collect_eos_ids(tok, target)
    dev = torch.device("cpu")
    n = 0
    while any(not s.done for s in seqs) and n < max_rounds:
        run_round(seqs, draft, target, gamma=gamma, temperature=temperature,
                  eos_ids=eos, device=dev, dtype=torch.float32, mode="spec")
        n += 1
    return seqs


def run_o5(mutants=(), *, gammas=(1, 3, 5), seeds=(0, 1, 2), eos=False,
           max_new_tokens=48, params=None, temperatures=(0.0, 1.0)) -> OracleResult:
    """Equivalence-preservation oracle for the batched path (P6.1).

    A `spec_kv_batch.run_round` round over the 6 prompts must emit exactly what 6
    independent `speculative_generate_kv` runs would, at matched seeds -- with the
    SAME fault operator active in both. This is not mutated-vs-clean (that is
    O1/O3); it asks whether the batched driver stays bit-identical to the
    single-sequence loop *even when the shared code path is broken*. The per-seq
    KV-cache design claims output equivalence "by construction"; O5 stress-tests
    that claim against all 17 operators. `killed` here means an operator broke
    batch/single equivalence -- a real `run_round` bug -- and the expected result
    is that NONE do."""
    params = params or {}
    draft, target, tok = make_fake_pair(eos=eos)
    n_runs = n_div = 0
    firsts: List[int] = []
    for temperature in temperatures:
        for gamma in gammas:
            for seed in seeds:
                # reference: single-sequence path, mutation active
                with faultlib.apply(*mutants, **params):
                    refs = [
                        speculative_generate_kv(p, draft, target, tok,
                                                make_cache=LengthOnlyCache, gamma=gamma,
                                                max_new_tokens=max_new_tokens,
                                                temperature=temperature, seed=seed).token_ids
                        for p in _PROMPTS
                    ]
                # batched path, same mutation
                seqs = [make_seq(f"s{i}", p, tok, device=torch.device("cpu"),
                                 max_new_tokens=max_new_tokens, seed=seed,
                                 make_cache=LengthOnlyCache)
                        for i, p in enumerate(_PROMPTS)]
                with faultlib.apply(*mutants, **params):
                    _drive_batch(seqs, draft, target, tok, gamma=gamma,
                                 temperature=temperature)
                for i in range(len(_PROMPTS)):
                    n_runs += 1
                    d = _first_divergence(seqs[i].token_ids, refs[i], tol_tail=gamma)
                    if d is not None:
                        n_div += 1
                        firsts.append(d)
    return OracleResult(oracle="O5", killed=n_div > 0, n_runs=n_runs, n_diverged=n_div,
                        first_divergence_tokens=firsts,
                        note="batched run_round vs N single-sequence speculative_generate_kv")


# --------------------------------------------------------------------------- #
# O4 -- structural invariants (always-on assertions)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _checked_math(violations: List[str]):
    """Wrap whatever adjusted_distribution / acceptance_probability are currently
    installed (mutated or not) with validity checks."""
    adj = _kv.adjusted_distribution
    acc = _kv.acceptance_probability

    def adj_checked(p_dm, p_tm):
        out = adj(p_dm, p_tm)
        s = float(out.sum())
        if not (abs(s - 1.0) < 1e-4):
            violations.append(f"adjusted_distribution sum={s:.6f} != 1")
        if float(out.min()) < -1e-6:
            violations.append(f"adjusted_distribution has negative mass {float(out.min()):.3e}")
        return out

    def acc_checked(p_dm_x, p_tm_x):
        a = acc(p_dm_x, p_tm_x)
        if not (-1e-9 <= a <= 1.0 + 1e-9):
            violations.append(f"acceptance_probability={a} out of [0,1]")
        return a

    saved = [(m, m.adjusted_distribution, m.acceptance_probability) for m in (_kv,)]
    try:
        _kv.adjusted_distribution = adj_checked
        _kv.acceptance_probability = acc_checked
        yield
    finally:
        for m, a1, a2 in saved:
            m.adjusted_distribution = a1
            m.acceptance_probability = a2


def _drive_once(prompt, draft, target, tok, gamma, seed, max_new_tokens, violations,
                temperature=0.0):
    """Reimplements the speculative_generate_kv loop so it can assert invariants
    after every round. Returns (token_ids, accept_reject_trace).

    temperature: 0.0 exercises the cache-length / control invariants; 1.0 makes
    the distributions non-one-hot so the wrapped adjusted_distribution /
    acceptance_probability validity checks actually bite."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    device = torch.device("cpu")
    ctx = encode_prompt(tok, prompt, device, True)
    committed = ctx[0].tolist()
    eos_ids = _rs.collect_eos_ids(tok, target)
    dcache, tcache = LengthOnlyCache(), LengthOnlyCache()
    dsync = tsync = 0
    out: List[int] = []
    trace: List[int] = []
    seen_eos = False
    while len(out) < max_new_tokens:
        g = min(gamma, max_new_tokens - len(out))
        step = speculative_step_kv(committed, draft, target, dcache, tcache, dsync, tsync, g,
                                   device=device, dtype=ctx.dtype, temperature=temperature,
                                   generator=gen)
        dsync, tsync = step.draft_synced, step.target_synced
        r = step.result
        if not (0 <= r.n_accepted <= g):
            violations.append(f"n_accepted={r.n_accepted} out of [0,{g}]")
        trace.append(r.n_accepted)

        emitted = r.new_token_ids
        hit = False
        for k, tid in enumerate(emitted):
            if tid in eos_ids:
                emitted = emitted[: k + 1]
                hit = True
                break
        if seen_eos and emitted:
            violations.append("token committed after EOS")
        out.extend(emitted)
        committed.extend(emitted)
        if hit:
            seen_eos = True
        if not hit and not seen_eos:
            if dcache.get_seq_length() != len(committed) - 1:
                violations.append(
                    f"draft cache len {dcache.get_seq_length()} != committed-1 {len(committed) - 1}")
            if tcache.get_seq_length() != len(committed) - 1:
                violations.append(
                    f"target cache len {tcache.get_seq_length()} != committed-1 {len(committed) - 1}")
        if hit:
            break
    return out, trace


def run_o4(mutants=(), *, gammas=(1, 3, 5), seeds=(0, 1, 2), eos=False,
           max_new_tokens=48, params=None) -> OracleResult:
    params = params or {}
    draft, target, tok = make_fake_pair(eos=eos)
    violations: List[str] = []
    n_runs = 0
    for gamma in gammas:
        for seed in seeds:
            for prompt in _PROMPTS:
                n_runs += 1
                for temp in (0.0, 1.0):
                    with faultlib.apply(*mutants, **params), _checked_math(violations):
                        t1, tr1 = _drive_once(prompt, draft, target, tok, gamma, seed,
                                              max_new_tokens, violations, temperature=temp)
                        t2, tr2 = _drive_once(prompt, draft, target, tok, gamma, seed,
                                              max_new_tokens, violations, temperature=temp)
                    if t1 != t2 or tr1 != tr2:
                        violations.append("non-deterministic under fixed seed")
    uniq = sorted(set(violations))
    return OracleResult(oracle="O4", killed=len(uniq) > 0, n_runs=n_runs,
                        n_diverged=len(violations), violations=uniq)


if __name__ == "__main__":
    print("clean baseline:")
    print(" ", run_o1())
    print(" ", run_o4())
