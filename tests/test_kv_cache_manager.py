"""P7 Track B -- BlockKVPool accounting, PrefixStore exact-match reuse, and
clone_dynamic_cache faithfulness. All hermetic (pure python + tiny tensors).
"""
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_cache_manager import (  # noqa: E402
    BlockKVPool,
    PrefixStore,
    clone_dynamic_cache,
)
from spec_oracles import run_o1_prefix  # noqa: E402


# --------------------------------------------------------------------------- #
# BlockKVPool
# --------------------------------------------------------------------------- #
def test_blocks_for_rounds_up():
    p = BlockKVPool(total_blocks=100, block_size=16)
    assert p.blocks_for(0) == 0
    assert p.blocks_for(1) == 1
    assert p.blocks_for(16) == 1
    assert p.blocks_for(17) == 2
    assert p.blocks_for(32) == 2


def test_acquire_release_free_consistency():
    p = BlockKVPool(total_blocks=10, block_size=16)
    assert p.free_blocks() == 10
    assert p.can_admit(16 * 3)
    p.acquire("a", 16 * 3)              # 3 blocks
    assert p.free_blocks() == 7
    p.acquire("b", 1)                   # 1 block
    assert p.free_blocks() == 6
    p.release("a")
    assert p.free_blocks() == 9
    p.release("b")
    assert p.free_blocks() == 10


def test_acquire_over_capacity_raises_memoryerror():
    p = BlockKVPool(total_blocks=2, block_size=16)
    with pytest.raises(MemoryError):
        p.acquire("x", 16 * 3)          # needs 3 blocks, only 2


def test_can_admit_boundary_one_block_pool():
    # feedback_smoke_test_boundary_conditions: hit the real cap, not a scaled one
    p = BlockKVPool(total_blocks=1, block_size=16)
    assert p.can_admit(16) is True
    assert p.can_admit(17) is False     # 2 blocks needed, 1 exists
    p.acquire("only", 16)
    assert p.can_admit(1) is False      # pool now full


def test_grow_returns_false_when_full():
    p = BlockKVPool(total_blocks=3, block_size=16)
    p.acquire("a", 16)                  # 1 block, 2 free
    assert p.grow("a", 16) is True      # -> 2 blocks, 1 free
    assert p.grow("a", 16) is True      # -> 3 blocks, 0 free
    assert p.grow("a", 16) is False     # nothing left
    assert p._used["a"] == 3            # unchanged by the failed grow


def test_pin_unpin_prefix_refcount():
    p = BlockKVPool(total_blocks=10, block_size=16)
    p.pin_prefix("k", 32)              # 2 shared blocks
    assert p.free_blocks() == 8
    p.pin_prefix("k", 32)              # same key, refcount 2, still 2 blocks
    assert p.free_blocks() == 8
    p.unpin_prefix("k", 32)           # refcount 1 -> still pinned
    assert p.free_blocks() == 8
    p.unpin_prefix("k", 32)           # refcount 0 -> blocks returned
    assert p.free_blocks() == 10


# --------------------------------------------------------------------------- #
# PrefixStore
# --------------------------------------------------------------------------- #
def _tiny_cache(seq_len, seed=0):
    torch.manual_seed(seed)
    c = DynamicCache()
    k = torch.randn(1, 2, seq_len, 4)
    v = torch.randn(1, 2, seq_len, 4)
    c.update(k, v, 0)
    return c


def test_longest_match_returns_longest_true_prefix():
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=8, min_prefix_tokens=3)
    st.put(list(range(10)), _tiny_cache(9), _tiny_cache(9, 1))
    st.put(list(range(5)), _tiny_cache(4), _tiny_cache(4, 1))

    n, e = st.longest_match(list(range(10)) + [99, 98])
    assert n == 10 and e is not None
    n2, e2 = st.longest_match(list(range(5)) + [42])
    assert n2 == 5 and e2 is not None


def test_longest_match_on_shared_prefix_not_whole_entry():
    # neither stored prompt is a whole prefix of the query; they only share a
    # leading run. longest_match must still find it (this is the shared
    # system-prompt case that exp2 exercises).
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=8, min_prefix_tokens=4)
    shared = list(range(12))
    st.put(shared + [100, 101], _tiny_cache(13), _tiny_cache(13, 1))
    n, e = st.longest_match(shared + [200, 201, 202])
    assert n == 12 and e is not None            # the 12-token shared run
    assert n < len(e.token_ids)                  # ... shorter than the stored entry


def test_seed_crops_stored_cache_to_the_shared_run():
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=8, min_prefix_tokens=4)
    shared = list(range(12))
    st.put(shared + [100, 101, 102, 103], _tiny_cache(15), _tiny_cache(15, 1))
    n, entry = st.longest_match(shared + [7, 7, 7])
    assert n == 12
    d, t, ds, ts = st.seed(entry, n)
    assert ds == ts == 11                        # matched - 1
    assert d.get_seq_length() == 11 and t.get_seq_length() == 11
    assert entry.draft_cache.get_seq_length() == 15   # stored entry untouched


def test_no_match_below_min_prefix_tokens():
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=8, min_prefix_tokens=8)
    st.put(list(range(5)), _tiny_cache(4), _tiny_cache(4, 1))   # too short -> not stored
    n, e = st.longest_match(list(range(5)))
    assert n == 0 and e is None


def test_put_is_idempotent():
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=8, min_prefix_tokens=3)
    st.put(list(range(10)), _tiny_cache(9), _tiny_cache(9, 1))
    st.put(list(range(10)), _tiny_cache(9), _tiny_cache(9, 1))
    assert len(st._entries) == 1
    assert sum(pool._refcount.values()) == 1


def test_evict_lru_drops_least_recently_used_and_unpins():
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=2, min_prefix_tokens=3)
    st.put([1, 2, 3, 4, 5, 6], _tiny_cache(5), _tiny_cache(5, 1))
    st.put([1, 2, 3, 4, 5, 7], _tiny_cache(5), _tiny_cache(5, 1))
    # touch the first so the second is LRU
    st.longest_match([1, 2, 3, 4, 5, 6])
    before = len(pool._refcount)
    st.put([1, 2, 3, 4, 5, 8], _tiny_cache(5), _tiny_cache(5, 1))   # forces eviction
    assert len(st._entries) == 2
    keys = {tuple(e.token_ids) for e in st._entries.values()}
    assert (1, 2, 3, 4, 5, 7) not in keys          # the LRU one went
    assert (1, 2, 3, 4, 5, 6) in keys
    assert len(pool._refcount) == before           # evicted entry was unpinned


def test_seed_returns_independent_clones_at_synced():
    pool = BlockKVPool(total_blocks=64)
    st = PrefixStore(pool, max_entries=8, min_prefix_tokens=3)
    st.put(list(range(9)), _tiny_cache(8), _tiny_cache(8, 1))
    _, entry = st.longest_match(list(range(9)))
    d, t, ds, ts = st.seed(entry)
    assert ds == ts == 8                            # len(token_ids) - 1
    assert d.get_seq_length() == 8 and t.get_seq_length() == 8
    assert d is not entry.draft_cache              # a real clone


# --------------------------------------------------------------------------- #
# clone_dynamic_cache faithfulness
# --------------------------------------------------------------------------- #
def test_clone_is_faithful():
    orig = _tiny_cache(6, seed=7)
    clone = clone_dynamic_cache(orig)
    # identical contents right after the clone
    assert torch.equal(orig.layers[0].keys, clone.layers[0].keys)
    assert torch.equal(orig.layers[0].values, clone.layers[0].values)
    # evolve both identically -> still bit-identical (proxy for equal next logits)
    torch.manual_seed(123)
    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    orig.update(k2.clone(), v2.clone(), 0)
    clone.update(k2.clone(), v2.clone(), 0)
    assert torch.equal(orig.layers[0].keys, clone.layers[0].keys)
    assert orig.get_seq_length() == clone.get_seq_length() == 7


def test_clone_is_independent():
    orig = _tiny_cache(5, seed=3)
    clone = clone_dynamic_cache(orig)
    clone.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 0)
    assert orig.get_seq_length() == 5           # original untouched
    assert clone.get_seq_length() == 6


# --------------------------------------------------------------------------- #
# oracle property
# --------------------------------------------------------------------------- #
def test_run_o1_prefix_property_holds():
    assert not run_o1_prefix().killed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
