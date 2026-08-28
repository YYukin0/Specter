"""
P6.5 Part 4 -- `specdiff`: a rule-based differential debugger for speculative
decoding.

When an oracle (O1/O3/O4 in src/spec_oracles.py) reports a divergence -- a real
bug, or a seeded mutant in blind-hunt mode -- specdiff localises it:

  1. round bisection      lockstep-rerun the suspect decoder and a trusted
                          reference in sampling mode, capture a per-round
                          structural signature (KV lengths, first position id,
                          n_accepted, accept/reject vector, emitted tokens,
                          committed-prefix hash); the first round whose signature
                          differs is the offending round R.
  2. round-R state dump    the two signatures at R plus the cached target verify
                          argmax vs a from-scratch recompute at the true prefix.
  3. mechanism classifier  rule-based signature match ->
                            UPSTREAM_KV_POS       KV length / position id / cached
                                                  verify logits disagree at R
                            SAMPLING_MATH         verify logits agree; accept/reject
                                                  vector or the resampled/bonus
                                                  token value disagrees
                            CONTROL_DESYNC        decisions agree; emitted-token
                                                  count disagrees (gamma / EOS /
                                                  accept-count desync)
                            BACKEND_NONDETERMINISM everything agrees yet tokens
                                                  diverge (only reachable on a
                                                  real backend; cf. arXiv:2607.17283)
  4. report               (round R, mechanism, evidence) -- Ekka-shaped, but
                          rule-based, local, free, speculative-decoding-specific.

Hermetic: the deterministic position-one-hot FakeModel from src/spec_oracles.py.
The same driver runs on real models once O2 lands.

Run:
  python src/specdiff.py --mutant kv_crop_off_by_one_minus
  python src/specdiff.py --blind-hunt
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rejection_sampling as _rs
import spec_faultlib as faultlib
import spec_kv as _kv
from spec_oracles import LengthOnlyCache, make_fake_pair

# NOTE: speculative_step_kv / collect_eos_ids are always reached through their
# module object (_kv.*, _rs.*) so faultlib's monkeypatches actually take effect
# inside this driver.

UPSTREAM_KV_POS = "UPSTREAM_KV_POS"
SAMPLING_MATH = "SAMPLING_MATH"
CONTROL_DESYNC = "CONTROL_DESYNC"
BACKEND_NONDETERMINISM = "BACKEND_NONDETERMINISM"
NO_DIVERGENCE = "NO_DIVERGENCE"

# signature-honest expected mechanism per operator (not the raw group label:
# force_accept perturbs the decision vector -> SAMPLING_MATH; eos_ignored changes
# the emitted count with decisions intact -> CONTROL_DESYNC).
OP_EXPECTED_MECHANISM: Dict[str, str] = {
    "resample_from_target": SAMPLING_MATH,
    # torch.multinomial silently renormalizes an unnormalized adjusted dist, so
    # this one is invisible to specdiff's trace signatures too -- only O4's
    # explicit sum-to-one assertion catches it (see p6_5_mutation_adequacy.json).
    "adjusted_no_renormalize": NO_DIVERGENCE,
    "adjusted_abs_not_relu": SAMPLING_MATH,
    "accept_ratio_inverted": SAMPLING_MATH,
    "accept_always": SAMPLING_MATH,
    "accept_strict": SAMPLING_MATH,
    "leniency_injected": SAMPLING_MATH,
    "bonus_token_from_draft": SAMPLING_MATH,
    "force_accept_first": SAMPLING_MATH,
    "eos_ignored_midblock": CONTROL_DESYNC,
    "kv_crop_off_by_one_minus": UPSTREAM_KV_POS,
    "kv_crop_off_by_one_plus": UPSTREAM_KV_POS,
    "kv_no_crop": UPSTREAM_KV_POS,
    "pos_id_off_by_one_plus": UPSTREAM_KV_POS,
    "pos_id_off_by_one_minus": UPSTREAM_KV_POS,
    "pos_id_frozen": UPSTREAM_KV_POS,
    # equivalent mutant -- specdiff is expected to report NO_DIVERGENCE
    "kv_crop_absolute_vs_relative": NO_DIVERGENCE,
}


# --------------------------------------------------------------------------- #
# instrumented lockstep driver
# --------------------------------------------------------------------------- #
@dataclass
class RoundState:
    idx: int
    draft_tokens: List[int]
    accept_reject: Tuple[int, ...]
    n_accepted: int
    emitted: Tuple[int, ...]
    draft_cache_len: int
    target_cache_len: int
    pos_ids_first: int
    cache_pos_calls: Tuple[Tuple[int, ...], ...]
    cached_verify_argmax0: Optional[int]
    recompute_argmax0: Optional[int]
    prefix_hash: str

    def struct_sig(self):
        return (tuple(self.draft_tokens), self.cache_pos_calls, self.draft_cache_len,
                self.target_cache_len, self.pos_ids_first, self.n_accepted,
                self.accept_reject, self.emitted, self.prefix_hash)


@dataclass
class Trace:
    rounds: List[RoundState] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)


def _hash(ids) -> str:
    return hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:12]


@contextlib.contextmanager
def _record_rows(sink: List[int]):
    """Record argmax of every distribution speculative_step_kv builds this round,
    in call order: gamma draft rows, then gamma+1 target verify rows, then
    (full-accept only) 1 draft bonus row."""
    orig = _kv.dist_from_logits

    def wrapped(logits_row, temperature):
        out = orig(logits_row, temperature)
        sink.append(int(out.argmax()))
        return out

    _kv.dist_from_logits = wrapped
    try:
        yield
    finally:
        _kv.dist_from_logits = orig


@contextlib.contextmanager
def _record_cache_pos(sink: List[Tuple[int, ...]]):
    """Record the position-id vector handed to every cached forward this round
    (draft chunk, each draft single-token step, target verify, bonus). This is
    the direct upstream signal for the M-POS operators."""
    orig = _kv._cache_position

    def wrapped(start, n, device):
        out = orig(start, n, device)
        sink.append(tuple(int(x) for x in out))
        return out

    _kv._cache_position = wrapped
    try:
        yield
    finally:
        _kv._cache_position = orig


def _run_instrumented(prompt, draft, target, tok, gamma, seed, max_new_tokens,
                      mutants=(), temperature=1.0) -> Trace:
    gen = torch.Generator()
    gen.manual_seed(seed)
    device = torch.device("cpu")
    ctx = _rs.encode_prompt(tok, prompt, device, True)
    committed = ctx[0].tolist()
    dcache, tcache = LengthOnlyCache(), LengthOnlyCache()
    dsync = tsync = 0
    tr = Trace()

    with faultlib.apply(*mutants):
        # collect_eos_ids is inside the mutation scope: eos_ignored_midblock
        # patches it to return an empty set.
        eos_ids = _rs.collect_eos_ids(tok, target)
        while len(tr.token_ids) < max_new_tokens:
            g = min(gamma, max_new_tokens - len(tr.token_ids))
            first_pos = tsync

            # from-scratch target argmax at the true committed prefix (greedy) --
            # ground truth for "is the cached verify logit corrupted?"
            rc = target(input_ids=torch.tensor([committed], device=device),
                        cache_position=torch.arange(len(committed))).logits[0]
            recompute_argmax0 = int(rc[-1].argmax())

            rows: List[int] = []
            cpos: List[Tuple[int, ...]] = []
            with _record_rows(rows), _record_cache_pos(cpos):
                step = _kv.speculative_step_kv(committed, draft, target, dcache, tcache,
                                              dsync, tsync, g, device=device, dtype=ctx.dtype,
                                              temperature=temperature, generator=gen, record=True)
            dsync, tsync = step.draft_synced, step.target_synced
            r = step.result
            # rows layout: [0:g] draft, [g:2g+1] target verify, [2g+1:] bonus
            cached_verify_argmax0 = rows[g] if len(rows) > g else None

            decisions = tuple(1 if p.get("accepted") else 0
                              for p in r.proposals if p.get("index") != "bonus")
            draft_toks = [int(p["token"]) for p in r.proposals if p.get("index") != "bonus"]

            emitted = list(r.new_token_ids)
            hit = False
            for k, tid in enumerate(emitted):
                if tid in eos_ids:
                    emitted = emitted[: k + 1]
                    hit = True
                    break
            committed.extend(emitted)
            tr.token_ids.extend(emitted)

            tr.rounds.append(RoundState(
                idx=len(tr.rounds),
                draft_tokens=draft_toks,
                accept_reject=decisions,
                n_accepted=r.n_accepted,
                emitted=tuple(emitted),
                draft_cache_len=dcache.get_seq_length(),
                target_cache_len=tcache.get_seq_length(),
                pos_ids_first=first_pos,
                cache_pos_calls=tuple(cpos),
                cached_verify_argmax0=cached_verify_argmax0,
                recompute_argmax0=recompute_argmax0,
                prefix_hash=_hash(tr.token_ids),
            ))
            if hit:
                break
    return tr


# --------------------------------------------------------------------------- #
# bisect + classify
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    offending_round: Optional[int]
    mechanism: str
    evidence: Dict
    diverged: bool


def bisect(ref: Trace, sus: Trace) -> Optional[int]:
    n = min(len(ref.rounds), len(sus.rounds))
    for i in range(n):
        if ref.rounds[i].struct_sig() != sus.rounds[i].struct_sig():
            return i
    if len(ref.rounds) != len(sus.rounds):
        return n
    return None


def classify(ref: Trace, sus: Trace, r: int) -> Report:
    if r >= len(ref.rounds) or r >= len(sus.rounds):
        # one run ran a different number of rounds -> a control-flow difference
        # (EOS handling, budget) rather than a math or cache one
        return Report(r, CONTROL_DESYNC,
                      {"rule": "round count differs (early stop / extra round)",
                       "ref_rounds": len(ref.rounds), "sus_rounds": len(sus.rounds)}, True)
    a, b = ref.rounds[r], sus.rounds[r]
    ev = {
        "round": r,
        "ref_cache": [a.draft_cache_len, a.target_cache_len],
        "sus_cache": [b.draft_cache_len, b.target_cache_len],
        "ref_pos_first": a.pos_ids_first, "sus_pos_first": b.pos_ids_first,
        "ref_decisions": list(a.accept_reject), "sus_decisions": list(b.accept_reject),
        "ref_n_accepted": a.n_accepted, "sus_n_accepted": b.n_accepted,
        "ref_emitted": list(a.emitted), "sus_emitted": list(b.emitted),
        "sus_cached_verify_argmax0": b.cached_verify_argmax0,
        "sus_recompute_argmax0": b.recompute_argmax0,
    }

    # Rule order matters: a corrupted KV / position feeding the verify forward
    # shows up as a wrong cached verify argmax or a shifted first position id,
    # and it also perturbs every downstream signal (decisions -> n_accepted ->
    # post-rollback cache length). So the *upstream* signals are checked first;
    # a bare cache-length mismatch with decisions and emitted tokens intact is
    # the last KV rule (the M-KV "invisible to output" case), and only then do
    # decision / token-value mismatches get read as sampling-math faults.

    # direct upstream signal for M-POS: the position-id vectors handed to the
    # cached forwards. Compare only the calls BOTH runs made (a trailing bonus
    # forward under a decision change would add one call downstream).
    ncp = min(len(a.cache_pos_calls), len(b.cache_pos_calls))
    if a.cache_pos_calls[:ncp] != b.cache_pos_calls[:ncp]:
        for i in range(ncp):
            if a.cache_pos_calls[i] != b.cache_pos_calls[i]:
                ev["ref_cache_pos_call"] = list(a.cache_pos_calls[i])
                ev["sus_cache_pos_call"] = list(b.cache_pos_calls[i])
                break
        ev["rule"] = "position-id vector fed to a cached forward differs at round R"
        return Report(r, UPSTREAM_KV_POS, ev, True)

    if (b.cached_verify_argmax0 is not None and b.recompute_argmax0 is not None
            and b.cached_verify_argmax0 != b.recompute_argmax0
            and a.cached_verify_argmax0 == a.recompute_argmax0):
        ev["rule"] = "cached target verify argmax disagrees with from-scratch recompute (stale/shifted KV feeding verify)"
        return Report(r, UPSTREAM_KV_POS, ev, True)

    if a.pos_ids_first != b.pos_ids_first:
        ev["rule"] = "first position id fed to the cached forward differs at round R"
        return Report(r, UPSTREAM_KV_POS, ev, True)

    # compare only the draft tokens BOTH runs examined -- the recorded list is
    # truncated at the acceptance stop, so a length difference is downstream of
    # the decisions, not a draft fault.
    shared = min(len(a.draft_tokens), len(b.draft_tokens))
    if list(a.draft_tokens[:shared]) != list(b.draft_tokens[:shared]):
        # first divergent round -> identical RNG state on entry, so the draft
        # would propose identical tokens unless what it *saw* changed (its KV or
        # the position ids fed to its forward). Sampling-math faults never touch
        # the draft proposal itself.
        ev["ref_draft_tokens"] = list(a.draft_tokens)
        ev["sus_draft_tokens"] = list(b.draft_tokens)
        ev["rule"] = "draft proposed different tokens from the same committed prefix (draft KV / position corruption)"
        return Report(r, UPSTREAM_KV_POS, ev, True)

    if a.accept_reject != b.accept_reject:
        ev["rule"] = "accept/reject vector differs while verify logits agree"
        return Report(r, SAMPLING_MATH, ev, True)

    if a.n_accepted == b.n_accepted and len(a.emitted) != len(b.emitted):
        ev["rule"] = "same decisions, different emitted-token count (gamma / EOS / accept-count desync)"
        return Report(r, CONTROL_DESYNC, ev, True)

    if a.emitted != b.emitted:
        ev["rule"] = "resampled / bonus token value differs (adjusted distribution or bonus provenance)"
        return Report(r, SAMPLING_MATH, ev, True)

    if [a.draft_cache_len, a.target_cache_len] != [b.draft_cache_len, b.target_cache_len]:
        ev["rule"] = "KV cache length differs at round R with decisions and emitted tokens intact (cache-management fault invisible to output)"
        return Report(r, UPSTREAM_KV_POS, ev, True)

    if a.prefix_hash != b.prefix_hash:
        ev["rule"] = "committed prefix differs with all structural state equal"
        return Report(r, BACKEND_NONDETERMINISM, ev, True)

    ev["rule"] = "bisect boundary artefact"
    return Report(r, CONTROL_DESYNC, ev, True)


def diagnose(prompt, mutants, *, gamma=3, seed=0, max_new_tokens=48,
             temperature=1.0) -> Report:
    eos = "eos_ignored_midblock" in mutants
    # phase_target close to the draft's phase (0.0) -> higher acceptance -> more
    # full-accept rounds, so bonus-token and leniency mutants actually fire.
    draft, target, tok = make_fake_pair(phase_target=0.3, eos=eos)
    ref = _run_instrumented(prompt, draft, target, tok, gamma, seed, max_new_tokens,
                            mutants=(), temperature=temperature)
    sus = _run_instrumented(prompt, draft, target, tok, gamma, seed, max_new_tokens,
                            mutants=mutants, temperature=temperature)
    r = bisect(ref, sus)
    if r is None:
        return Report(None, NO_DIVERGENCE, {"note": "traces identical"}, False)
    return classify(ref, sus, r)


# --------------------------------------------------------------------------- #
# blind mutant hunt
# --------------------------------------------------------------------------- #
def blind_hunt(n=40, gamma=3, prompts=None) -> Dict:
    prompts = prompts or ["a longer prompt here", "hello world", "the quick brown fox"]
    ops = [o for o in faultlib.names() if OP_EXPECTED_MECHANISM.get(o) != NO_DIVERGENCE]
    rng = random.Random(0)
    hits = 0
    manifested = 0        # runs where the mutant produced any trace divergence
    manifested_hits = 0   # ... and specdiff named the right mechanism
    confusion: Dict[str, Dict[str, int]] = {}
    per_op: Dict[str, List[bool]] = {}
    for i in range(n):
        name = rng.choice(ops)
        prompt = rng.choice(prompts)
        expected = OP_EXPECTED_MECHANISM[name]
        got = diagnose(prompt, (name,), gamma=gamma, seed=i).mechanism
        confusion.setdefault(expected, {}).setdefault(got, 0)
        confusion[expected][got] += 1
        per_op.setdefault(name, []).append(got == expected)
        hits += (got == expected)
        if got != NO_DIVERGENCE:
            manifested += 1
            manifested_hits += (got == expected)
    return {
        "n": n,
        "accuracy": hits / n if n else 0.0,
        # the honest headline: when the mutant actually perturbs the trace and
        # specdiff commits to a mechanism, how often is that mechanism right?
        # (misses are all "mild mutant, no divergence on this seed", never a
        # wrong mechanism.)
        "mechanism_precision_when_manifested": manifested_hits / manifested if manifested else 0.0,
        "n_manifested": manifested,
        "confusion_expected_x_predicted": confusion,
        "per_operator_hit_rate": {k: sum(v) / len(v) for k, v in sorted(per_op.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", action="append", default=[])
    ap.add_argument("--prompt", default="a longer prompt here")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind-hunt", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="blind-hunt trial count")
    ap.add_argument("--persist", action="store_true",
                    help="write blind-hunt result to results/p6_5_specdiff_blind_hunt.json")
    args = ap.parse_args()

    if args.blind_hunt:
        out = blind_hunt(n=args.n, gamma=args.gamma)
        print(json.dumps(out, indent=2))
        if args.persist:
            p = Path(__file__).resolve().parent.parent / "results" / "p6_5_specdiff_blind_hunt.json"
            p.write_text(json.dumps(out, indent=2))
            print(f"wrote {p}")
        return
    rep = diagnose(args.prompt, tuple(args.mutant), gamma=args.gamma, seed=args.seed)
    print(json.dumps({
        "mutants": args.mutant,
        "offending_round": rep.offending_round,
        "mechanism": rep.mechanism,
        "diverged": rep.diverged,
        "evidence": rep.evidence,
    }, indent=2))


if __name__ == "__main__":
    main()
