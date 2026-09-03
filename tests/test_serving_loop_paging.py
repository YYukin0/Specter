"""P7 Track B -- SpecServer with the block KV pool + exact-prefix store.

Hermetic (make_fake_pair + LengthOnlyCache). Pins:
  * pool + prefix store ON emits byte-identical token streams vs all-off
    (capacity accounting and eager/clone prefill must not perturb a single
    output token);
  * a pool too small for every queued request at once still completes them
    all -- admission just defers, it never drops (坑34);
  * a workload where request B's prompt contains request A's prompt as a
    leading run actually reuses the stored KV: prefill_skip_ratio > 0.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from serving_loop import ServeConfig, SpecServer  # noqa: E402
from spec_oracles import LengthOnlyCache, _first_divergence, make_fake_pair  # noqa: E402


class _CharTok:
    """One deterministic token per prompt character, so a longer prompt that
    starts with a shorter one tokenises to a strict superset -- the exact-prefix
    store's reuse condition."""
    eos_token_id = None
    pad_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt"):
        content = msgs[0]["content"]
        return torch.tensor([[1 + (i % 50) for i in range(len(content))]])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


BASE = "aaaaaaaaaaaa"          # 12 tokens
PROMPTS = [BASE, BASE + "bbb", BASE + "bbbccc", "zzzzzzzzzz", "yyyyyyyyyyyyyy"]


def _server(*, pool_blocks=0, prefix=False, max_active=8, tok=None):
    draft, target, _tok = make_fake_pair()
    cfg = ServeConfig(gamma=3, temperature=0.0, max_new_tokens=24,
                      max_active=max_active, make_cache=LengthOnlyCache,
                      breaker_on=False, controller="fixed",
                      kv_total_blocks=pool_blocks, kv_block_size=16,
                      prefix_cache=prefix, apply_chat_template=True)
    return SpecServer(draft, target, tok or _tok, cfg)


def _run(srv, prompts):
    for i, p in enumerate(prompts):
        srv.submit(p, req_id=f"q{i}", seed=1000 + i)
    srv.run_until_idle(max_rounds=5000)
    return {f"q{i}": srv.poll(f"q{i}").token_ids for i in range(len(prompts))}


def test_pool_and_prefix_are_output_identical_to_all_off():
    off = _run(_server(tok=_CharTok()), PROMPTS)
    on = _run(_server(pool_blocks=64, prefix=True, tok=_CharTok()), PROMPTS)
    assert set(off) == set(on)
    for k in off:
        assert _first_divergence(off[k], on[k], tol_tail=0) is None, \
            f"pool+prefix changed output for {k}"


def test_small_pool_defers_but_completes_every_request():
    # 4 blocks * 16 = 64 token capacity; the 5 prompts together need more, so
    # admission must queue some and drain them as others finish.
    srv = _server(pool_blocks=4, prefix=False, max_active=8, tok=_CharTok())
    out = _run(srv, PROMPTS)
    assert len(srv.results()) == len(PROMPTS)
    assert all(len(v) > 0 for v in out.values())
    assert srv.pool.free_blocks() == srv.pool.total_blocks   # all released


def test_prefix_reuse_skips_shared_prefill_work():
    srv = _server(pool_blocks=64, prefix=True, tok=_CharTok())
    _run(srv, [BASE, BASE + "bbb", BASE + "bbbccc"])
    # q1 and q2 each start with an already-cached shorter prompt
    assert srv.prefill_skip_ratio > 0.0
    assert srv.prefill_tokens_skipped >= 12


def test_prefix_reuse_on_shared_leading_run_not_whole_prompt():
    # the realistic case: a common ~system-prompt run, then each request
    # diverges. No prompt is a whole prefix of another, but the shared run is
    # still reused (longest_match finds the common leading tokens, seed crops).
    sys_run = "a" * 20
    prompts = [sys_run + "bbbb", sys_run + "cccc", sys_run + "dddd", sys_run + "eeee"]
    srv = _server(pool_blocks=64, prefix=True, tok=_CharTok())
    _run(srv, prompts)
    assert srv.prefill_skip_ratio > 0.0
    # 3 later requests each skip ~19 shared tokens
    assert srv.prefill_tokens_skipped >= 3 * 15

    # and output is still byte-identical to prefix OFF
    off = _run(_server(pool_blocks=64, prefix=False, tok=_CharTok()), prompts)
    on = _run(_server(pool_blocks=64, prefix=True, tok=_CharTok()), prompts)
    for k in off:
        assert _first_divergence(off[k], on[k], tol_tail=0) is None


def test_prefix_off_means_no_skips():
    srv = _server(pool_blocks=64, prefix=False, tok=_CharTok())
    _run(srv, [BASE, BASE + "bbb", BASE + "bbbccc"])
    assert srv.prefill_tokens_skipped == 0
    assert srv.prefill_skip_ratio == 0.0


def test_pool_accounting_invariant_holds_every_round():
    # B3 O4-style invariant: free_blocks never goes negative and the pinned +
    # in-use block count never exceeds the pool, at every step of a run that
    # forces queueing (small pool, many prompts).
    srv = _server(pool_blocks=6, prefix=True, max_active=8, tok=_CharTok())
    for i, p in enumerate(PROMPTS):
        srv.submit(p, req_id=f"q{i}", seed=1000 + i)
    while srv.active or srv.pending:
        srv.step()
        pool = srv.pool
        assert pool.free_blocks() >= 0
        assert sum(pool._used.values()) + pool._shared_blocks <= pool.total_blocks
    assert len(srv.results()) == len(PROMPTS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
