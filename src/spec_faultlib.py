"""
P6.5 Part 2 -- fault library: speculative-decoding-specific mutation operators.

Mutation testing as an idea is from the 1970s. What is (per the four literature
re-reviews in notes/deployment-depth-plan_2026-08-28.md sec 7 C6) *not* published
is a reusable, framework-agnostic catalogue of **speculative-decoding-specific**
fault operators plus an oracle stack that says which class of bug each oracle can
and cannot see. This module is the operator catalogue; src/spec_oracles.py is the
oracle stack; the mutation-adequacy matrix they produce
(results/p6_5_mutation_adequacy.json) is the citable finding.

Each operator is a context manager that monkeypatches `rejection_sampling`
and/or `spec_kv` for its scope and restores on exit. They compose via `apply()`.
`Injection` (src/rejection_sampling.py) already covered two of these (坑2 bonus,
forced-accept); the rest are new.

Groups:
  M-SAMPLE  rejection-sampling math (accept ratio, adjusted distribution, bonus)
  M-CTRL    speculative control flow (EOS handling, forced accept)
  M-KV      KV-cache management (crop off-by-one, absolute vs relative, no crop)
  M-POS     position ids fed to the cached forwards

Batch-only operators (M-POS mask_* / left-pad drift) need spec_kv_batch.py (P6.1)
and are registered as `requires="P6.1"` placeholders, not yet active.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Dict, List

import torch

import rejection_sampling as _rs
import spec_kv as _kv
import spec_kv_batch as _kvb   # P6.1 batched decoder -- operators must reach it too

# modules that did `from rejection_sampling import <pure helper>` -- a patch must
# hit every binding, not just the origin module. `_kvb` re-imports dist_from_logits
# / _sample (used by its degraded-mode step); its speculative path goes through
# spec_kv.speculative_step_kv, so the M-KV / M-POS patches on `_kv` already cover
# it, but the math-helper bindings it copied need patching here.
_MATH_MODULES = (_rs, _kv, _kvb)


# --------------------------------------------------------------------------- #
# patching primitives
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _patch(modules, attr: str, new_value):
    saved = [(m, getattr(m, attr)) for m in modules if hasattr(m, attr)]
    try:
        for m, _ in saved:
            setattr(m, attr, new_value)
        yield
    finally:
        for m, old in saved:
            setattr(m, attr, old)


@contextlib.contextmanager
def _wrap_math(attr: str, make_new: Callable[[Callable], Callable]):
    """Replace pure helper `attr` in _MATH_MODULES with make_new(original)."""
    original = getattr(_rs, attr)
    with _patch(_MATH_MODULES, attr, make_new(original)):
        yield


@contextlib.contextmanager
def _inject(**injection_fields):
    """Route both step functions through an rejection_sampling.Injection with the
    given fields forced on (merging with any injection the caller passed)."""
    orig_rs_step = _rs.speculative_step
    orig_kv_step = _kv.speculative_step_kv

    def _merge(passed):
        base = passed or _rs.Injection()
        return _rs.Injection(
            bonus_from_draft=injection_fields.get("bonus_from_draft", base.bonus_from_draft),
            force_accept_index=injection_fields.get("force_accept_index", base.force_accept_index),
        )

    def wrap(orig):
        def wrapped(*args, **kw):
            kw["injection"] = _merge(kw.get("injection"))
            return orig(*args, **kw)
        return wrapped

    orig_kvb_step = getattr(_kvb, "speculative_step_kv", None)
    try:
        _rs.speculative_step = wrap(orig_rs_step)
        _kv.speculative_step_kv = wrap(orig_kv_step)
        if orig_kvb_step is not None:                 # batched path's own binding
            _kvb.speculative_step_kv = wrap(orig_kvb_step)
        yield
    finally:
        _rs.speculative_step = orig_rs_step
        _kv.speculative_step_kv = orig_kv_step
        if orig_kvb_step is not None:
            _kvb.speculative_step_kv = orig_kvb_step


# --------------------------------------------------------------------------- #
# M-SAMPLE
# --------------------------------------------------------------------------- #
def _m_resample_from_target():
    # resample on rejection from raw p_TM instead of norm(max(0, p_TM - p_DM)).
    return _wrap_math("adjusted_distribution", lambda _orig: (lambda p_dm, p_tm: p_tm.clone()))


def _m_adjusted_no_renormalize():
    def make(_orig):
        def f(p_dm, p_tm):
            return torch.clamp(p_tm - p_dm, min=0.0)  # never divided by its sum
        return f
    return _wrap_math("adjusted_distribution", make)


def _m_adjusted_abs_not_relu():
    def make(_orig):
        def f(p_dm, p_tm):
            r = (p_tm - p_dm).abs()
            s = r.sum()
            return r / s if s > 1e-12 else p_tm.clone()
        return f
    return _wrap_math("adjusted_distribution", make)


def _m_accept_ratio_inverted():
    def make(_orig):
        def f(p_dm_x, p_tm_x):
            p_dm_x, p_tm_x = float(p_dm_x), float(p_tm_x)
            if p_tm_x <= 0.0:
                return 1.0
            return min(1.0, p_dm_x / p_tm_x)  # ratio the wrong way up
        return f
    return _wrap_math("acceptance_probability", make)


def _m_accept_always():
    return _wrap_math("acceptance_probability", lambda _orig: (lambda p_dm_x, p_tm_x: 1.0))


def _m_accept_strict():
    # accept iff p_TM >= p_DM; drop the random draw entirely (r < 1.0 is always
    # true, r < 0.0 never) -> deterministic, biased acceptance.
    def make(_orig):
        def f(p_dm_x, p_tm_x):
            return 1.0 if float(p_tm_x) >= float(p_dm_x) else 0.0
        return f
    return _wrap_math("acceptance_probability", make)


def _m_leniency(l: float = 1.05):
    def make(orig):
        def f(p_dm_x, p_tm_x):
            return min(1.0, l * orig(p_dm_x, p_tm_x))
        return f
    return _wrap_math("acceptance_probability", make)


def _m_bonus_from_draft():
    return _inject(bonus_from_draft=True)


# --------------------------------------------------------------------------- #
# M-CTRL
# --------------------------------------------------------------------------- #
def _m_eos_ignored_midblock():
    # collect_eos_ids -> empty set: the generate loop never truncates mid-chunk,
    # so tokens are emitted past where plain autoregressive decoding would stop.
    return _patch(_MATH_MODULES, "collect_eos_ids", lambda tokenizer, model: set())


def _m_force_accept_first():
    return _inject(force_accept_index=0)


# --------------------------------------------------------------------------- #
# M-KV  (spec_kv only)
# --------------------------------------------------------------------------- #
def _m_kv_crop_minus_one():
    orig = _kv._crop_to
    return _patch((_kv,), "_crop_to", lambda cache, n: orig(cache, max(0, n - 1)))


def _m_kv_crop_plus_one():
    orig = _kv._crop_to
    return _patch((_kv,), "_crop_to", lambda cache, n: orig(cache, n + 1))


def _m_kv_crop_absolute():
    # drive DynamicCache.crop with a positive (legacy "absolute length") arg
    def f(cache, n):
        cur = cache.get_seq_length()
        if n >= cur:
            return
        cache.crop(n)  # positive -> deprecated absolute-length semantics
    return _patch((_kv,), "_crop_to", f)


def _m_kv_crop_noop():
    return _patch((_kv,), "_crop_to", lambda cache, n: None)


# --------------------------------------------------------------------------- #
# M-POS  (spec_kv only)
# --------------------------------------------------------------------------- #
def _m_pos_shift_plus_one():
    def f(start, n, device):
        return torch.arange(start + 1, start + 1 + n, device=device, dtype=torch.long)
    return _patch((_kv,), "_cache_position", f)


def _m_pos_shift_minus_one():
    def f(start, n, device):
        return torch.arange(max(0, start - 1), max(0, start - 1) + n, device=device, dtype=torch.long)
    return _patch((_kv,), "_cache_position", f)


def _m_pos_frozen():
    def f(start, n, device):
        return torch.full((n,), start, device=device, dtype=torch.long)
    return _patch((_kv,), "_cache_position", f)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
MUTATORS: Dict[str, Callable[[], "contextlib.AbstractContextManager"]] = {
    # M-SAMPLE
    "resample_from_target": _m_resample_from_target,
    "adjusted_no_renormalize": _m_adjusted_no_renormalize,
    "adjusted_abs_not_relu": _m_adjusted_abs_not_relu,
    "accept_ratio_inverted": _m_accept_ratio_inverted,
    "accept_always": _m_accept_always,
    "accept_strict": _m_accept_strict,
    "leniency_injected": _m_leniency,
    "bonus_token_from_draft": _m_bonus_from_draft,
    # M-CTRL
    "eos_ignored_midblock": _m_eos_ignored_midblock,
    "force_accept_first": _m_force_accept_first,
    # M-KV
    "kv_crop_off_by_one_minus": _m_kv_crop_minus_one,
    "kv_crop_off_by_one_plus": _m_kv_crop_plus_one,
    "kv_crop_absolute_vs_relative": _m_kv_crop_absolute,
    "kv_no_crop": _m_kv_crop_noop,
    # M-POS
    "pos_id_off_by_one_plus": _m_pos_shift_plus_one,
    "pos_id_off_by_one_minus": _m_pos_shift_minus_one,
    "pos_id_frozen": _m_pos_frozen,
}

GROUP: Dict[str, str] = {
    "resample_from_target": "M-SAMPLE",
    "adjusted_no_renormalize": "M-SAMPLE",
    "adjusted_abs_not_relu": "M-SAMPLE",
    "accept_ratio_inverted": "M-SAMPLE",
    "accept_always": "M-SAMPLE",
    "accept_strict": "M-SAMPLE",
    "leniency_injected": "M-SAMPLE",
    "bonus_token_from_draft": "M-SAMPLE",
    "eos_ignored_midblock": "M-CTRL",
    "force_accept_first": "M-CTRL",
    "kv_crop_off_by_one_minus": "M-KV",
    "kv_crop_off_by_one_plus": "M-KV",
    "kv_crop_absolute_vs_relative": "M-KV",
    "kv_no_crop": "M-KV",
    "pos_id_off_by_one_plus": "M-POS",
    "pos_id_off_by_one_minus": "M-POS",
    "pos_id_frozen": "M-POS",
}

# Operators catalogued but not active, with WHY (updated after P6.1 landed).
#
# P6.1 (src/spec_kv_batch.py) took the EQSPEC rectangular invariant to its limit:
# per-sequence KV caches, no shared ragged batch tensor. That design decision
# makes the mask/pad-drift operators *architecturally inexpressible* on this
# codebase -- there is no attention mask, no left-pad, and no cross-sequence
# position-id tensor to corrupt. Their inexpressibility here IS the sec 7 C3
# finding (notes/project_plan_v9.md 坑19): the per-seq path buys output
# equivalence by construction, at the cost of kernel batching.
#
#   mask_leak_future / mask_left_pad_drift
#       Only expressible against the shared-tensor masking route in
#       src/spec_batch.py (M5[A], no KV cache). And the deterministic FakeModel
#       ignores position_ids (it keys logits off cache_position), so reproducing
#       the arXiv:2510.22876 BSP signature needs a RoPE-sensitive *real* model ->
#       it is a P6.4 real-model demo, not part of the hermetic matrix.
#   kv_swap_draft_target_crop / accept_count_desync
#       Expressible on SeqState, but invisible to output-equivalence oracles
#       (O1/O3/O5): they corrupt per-stream telemetry / cache bookkeeping without
#       changing the tokens. Catching them needs a telemetry-consistency oracle
#       (O6), which is the next P6.5 increment.
#   gamma_off_by_one_verify
#       Needs a step-level rewrite of the verify loop, not a monkeypatch.
DEFERRED = {
    "mask_leak_future": "inexpressible on per-seq caches; real-model spec_batch.py only",
    "mask_left_pad_drift": "arXiv:2510.22876 BSP signature; needs RoPE real model (P6.4)",
    "kv_swap_draft_target_crop": "expressible, invisible to O1/O3/O5; needs O6 telemetry oracle",
    "accept_count_desync": "expressible, invisible to O1/O3/O5; needs O6 telemetry oracle",
    "gamma_off_by_one_verify": "step-level rewrite, not a monkeypatch",
}


def names() -> List[str]:
    return list(MUTATORS)


@contextlib.contextmanager
def apply(*mutant_names: str, **params):
    """Install one or more mutation operators for the duration of the block.
    `apply()` with no names is a no-op (the clean baseline)."""
    with contextlib.ExitStack() as stack:
        for name in mutant_names:
            if name not in MUTATORS:
                raise KeyError(f"unknown mutator {name!r}; known: {sorted(MUTATORS)}")
            factory = MUTATORS[name]
            kw = params.get(name, {})
            stack.enter_context(factory(**kw) if kw else factory())
        yield
