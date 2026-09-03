"""
P7 Track B -- block-structured KV accounting + exact-prefix reuse.

SCOPE (this module does NOT replace the attention kernel). Computation still runs
each sequence's own `transformers.DynamicCache` through `speculative_step_kv`
(src/spec_kv.py) -- already correct, already tested. What is added:

  1. `BlockKVPool` -- a fixed block pool with a free count + refcounted shared
     prefixes, so `SpecServer` admission is driven by real KV capacity instead of
     a hard-coded `max_active`.
  2. `PrefixStore` -- `hash(prefix token ids) -> stored DynamicCache prefix`. A
     new request that shares a leading token run with a cached one clones that
     KV and skips the shared prefill (`synced` is set past the shared run).

PagedAttention's kernel side (gathering K/V from non-contiguous blocks) is
deliberately NOT built -- see docs/engineering-notes/11-block-kv-and-prefix-reuse.md:
transformers' own `PagedAttentionCache` has no `crop` (speculative rollback
unsupported), the MPS FlexAttention paged path has a correctness bug since
2026-05 (PyTorch #182593), and `paged|sdpa` materialises the mask, is O(sum tok^2),
and likewise has no crop / no speculative support.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def clone_dynamic_cache(cache) -> Any:
    """Deep-copy a DynamicCache (or a length-only fake) so the clone yields
    bit-identical next-token logits. Verified in
    tests/test_kv_cache_manager.py::test_clone_is_faithful (坑35: relies on
    `copy.deepcopy` reproducing every per-layer key/value tensor exactly)."""
    return copy.deepcopy(cache)


# --------------------------------------------------------------------------- #
# block pool
# --------------------------------------------------------------------------- #
class BlockKVPool:
    """Fixed pool of KV blocks. Per-request usage is tracked in whole blocks;
    shared prefixes are pinned separately and refcounted so an in-use prefix is
    never reclaimed (坑34)."""

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._used: Dict[str, int] = {}          # req_id -> n_blocks
        self._refcount: Dict[str, int] = {}      # prefix_key -> int
        self._shared_blocks = 0

    def blocks_for(self, n_tokens: int) -> int:
        return (max(0, n_tokens) + self.block_size - 1) // self.block_size

    def free_blocks(self) -> int:
        return self.total_blocks - sum(self._used.values()) - self._shared_blocks

    def can_admit(self, n_tokens: int) -> bool:
        return self.blocks_for(n_tokens) <= self.free_blocks()

    def acquire(self, req_id: str, n_tokens: int) -> None:
        b = self.blocks_for(n_tokens)
        if b > self.free_blocks():
            raise MemoryError(f"pool full: need {b}, free {self.free_blocks()}")
        self._used[req_id] = self._used.get(req_id, 0) + b

    def grow(self, req_id: str, extra_tokens: int) -> bool:
        """Grow a request by whole blocks as its sequence extends. Returns False
        (without mutating) if the pool cannot cover the growth."""
        cur_blocks = self._used.get(req_id, 0)
        cur_tok = cur_blocks * self.block_size
        need = self.blocks_for(cur_tok + extra_tokens) - cur_blocks
        if need <= 0:
            return True
        if need > self.free_blocks():
            return False
        self._used[req_id] = cur_blocks + need
        return True

    def release(self, req_id: str) -> None:
        self._used.pop(req_id, None)

    def pin_prefix(self, prefix_key: str, n_tokens: int) -> None:
        if prefix_key not in self._refcount:
            self._shared_blocks += self.blocks_for(n_tokens)
            self._refcount[prefix_key] = 0
        self._refcount[prefix_key] += 1

    def unpin_prefix(self, prefix_key: str, n_tokens: int) -> None:
        if prefix_key in self._refcount:
            self._refcount[prefix_key] -= 1
            if self._refcount[prefix_key] <= 0:
                self._shared_blocks -= self.blocks_for(n_tokens)
                del self._refcount[prefix_key]


# --------------------------------------------------------------------------- #
# exact-prefix store
# --------------------------------------------------------------------------- #
@dataclass
class PrefixEntry:
    token_ids: tuple
    draft_cache: object
    target_cache: object
    synced: int           # = len(token_ids) - 1  (rectangular invariant)
    key: str
    last_use: int = 0


class PrefixStore:
    """`hash(leading token ids) -> cloned DynamicCache prefix`. Only exact
    leading-run matches (same tokens from position 0, same offset) are reused, so
    the shared KV is bit-identical to a from-scratch prefill (坑33: RoPE / causal
    attention make a leading run's KV position-independent of what follows)."""

    def __init__(self, pool: BlockKVPool, max_entries: int = 32, min_prefix_tokens: int = 8):
        self.pool = pool
        self.max_entries = max_entries
        self.min_prefix_tokens = min_prefix_tokens
        self._entries: Dict[str, PrefixEntry] = {}
        self._clock = 0

    @staticmethod
    def key_for(token_ids) -> str:
        h = hashlib.sha1()
        for t in token_ids:
            h.update(int(t).to_bytes(4, "little", signed=True))
        return h.hexdigest()

    def longest_match(self, token_ids) -> Tuple[int, Optional[PrefixEntry]]:
        """Longest common leading token run between `token_ids` and any stored
        entry. Returns (matched_len, PrefixEntry|None) with
        matched_len <= min(len(entry), len(token_ids)); the stored cache is
        cropped down to that run on `seed`. A whole stored prompt being a strict
        prefix of the query is just the special case matched_len == len(entry)
        (坑33: a leading run's KV is independent of what follows, given the same
        offset + contiguous position ids + full causal mask)."""
        ids = [int(t) for t in token_ids]
        best: Tuple[int, Optional[PrefixEntry]] = (0, None)
        for e in self._entries.values():
            cap = min(len(e.token_ids), len(ids))
            m = 0
            while m < cap and e.token_ids[m] == ids[m]:
                m += 1
            if m >= self.min_prefix_tokens and m > best[0]:
                best = (m, e)
        if best[1] is not None:
            self._clock += 1
            best[1].last_use = self._clock
        return best

    def put(self, token_ids, draft_cache, target_cache) -> None:
        """Store the prompt-prefix KV. `draft_cache`/`target_cache` must be at
        length len(token_ids) - 1 (post-prefill rectangular invariant)."""
        if len(token_ids) < self.min_prefix_tokens:
            return
        key = self.key_for(token_ids)
        if key in self._entries:
            return
        if len(self._entries) >= self.max_entries:
            self._evict_lru()
        e = PrefixEntry(tuple(int(t) for t in token_ids),
                        clone_dynamic_cache(draft_cache),
                        clone_dynamic_cache(target_cache),
                        len(token_ids) - 1, key)
        self._entries[key] = e
        self.pool.pin_prefix(key, len(token_ids))

    def seed(self, entry: PrefixEntry, matched: Optional[int] = None):
        """Fresh caches cloned from a stored prefix, cropped to the common run.

        `matched` is the number of shared leading prompt tokens (from
        `longest_match`); default is the full stored entry. Both caches are
        returned at length `matched - 1` (rectangular invariant -- the last
        shared token stays as the first pending input), with
        draft_synced == target_synced == matched - 1. Cropping a stored cache to
        a shorter shared run only discards later positions, so the retained KV is
        still bit-identical to a from-scratch prefill (坑33)."""
        full = entry.synced + 1
        if matched is None or matched > full:
            matched = full
        target_len = matched - 1
        d = clone_dynamic_cache(entry.draft_cache)
        t = clone_dynamic_cache(entry.target_cache)
        if target_len < entry.synced:
            d.crop(-(entry.synced - target_len))
            t.crop(-(entry.synced - target_len))
        return d, t, target_len, target_len

    def _evict_lru(self) -> None:
        k = min(self._entries, key=lambda kk: self._entries[kk].last_use)
        e = self._entries.pop(k)
        self.pool.unpin_prefix(e.key, len(e.token_ids))
