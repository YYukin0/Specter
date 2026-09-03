# Block-structured KV accounting and exact-prefix reuse — without touching the kernel

**Task:** Pillar 8 (支柱8) Track B — make `SpecServer` admission driven by real
KV capacity instead of a hard-coded `max_active`, and let a new request that
shares a leading token run with a cached one skip the shared prefill.
**Pitfalls:** 33, 34, 35.
**Code:** `src/kv_cache_manager.py`, `src/serving_loop.py` (admission path),
`src/verify_p7_2_paging.py`, `tests/test_kv_cache_manager.py`,
`tests/test_serving_loop_paging.py`, `src/spec_oracles.py::run_o1_prefix`,
`results/p7_2_paging.json`.

## Scope: this does not replace the attention kernel

Computation still runs each sequence's own `transformers.DynamicCache` through
`speculative_step_kv` (note 02) — the code that's already correct and already
tested. What this track adds is *bookkeeping* on top:

1. **`BlockKVPool`** — a fixed pool of KV blocks with a free count and
   refcounted shared prefixes, so admission asks "is there capacity?" rather
   than "are we under `max_active`?".
2. **`PrefixStore`** — `hash(leading token ids) -> cloned DynamicCache prefix`.
   A request whose prompt begins with an already-cached prompt clones that KV
   and sets `synced` past the shared run, skipping that much prefill.

The PagedAttention *kernel* side — gathering K/V from non-contiguous blocks —
is deliberately **not built**. Three concrete blockers on this stack:

- `transformers`' own `PagedAttentionCache` has **no `crop`**, so speculative
  rollback is unsupported — a non-starter for a speculative decoder.
- The MPS FlexAttention paged path has had a correctness bug since 2026-05
  (PyTorch #182593).
- `paged|sdpa` materializes the attention mask, is O(sum of tokens squared),
  and likewise has no `crop` / no speculative support.

So a real paged kernel here would mean forking one of those and fixing
speculative rollback in it — a project, not a track. The bookkeeping is the
part that's both tractable and independently useful (it's what decides
admission and prefix reuse regardless of how attention is computed).

## `BlockKVPool`

Per-request usage is tracked in whole blocks (`block_size = 16` tokens);
`blocks_for(n) = ceil(n / 16)`. `free_blocks()` is
`total - sum(per-request) - shared_prefix_blocks`. `can_admit(n)` gates
admission; `acquire` raises `MemoryError` if it's called past capacity (it
never should be — `_admit` checks `can_admit` first and `break`s the admission
loop otherwise, leaving the request queued for a later round).

Shared prefixes are **pinned and refcounted** separately from per-request
blocks (`pin_prefix` / `unpin_prefix`). A prefix block is only returned to the
pool when its refcount hits zero — while any live sequence was seeded from that
prefix, or the `PrefixStore` still holds the entry, it can't be reclaimed
(Pitfall 34: an evicted-but-still-referenced prefix block is a silent
use-after-free of KV state).

## `PrefixStore`

`key_for` is SHA-1 over the little-endian int32 token bytes. `longest_match`
scans stored entries and returns the **longest common leading run** between the
incoming prompt and any entry (token-by-token to the first mismatch), as long as
it's at least `min_prefix_tokens`. A stored prompt that is wholly a prefix of
the new one is just the special case `matched == len(entry)`. On a hit, `seed`
hands back **deep-copied** draft and target caches; if the match is shorter than
the stored entry it `crop`s the clones down to `matched - 1` first (the same
relative `crop(-n)` the speculative rollback uses). The synced offset is
`matched - 1` (rectangular invariant), and the serving loop prefills only the
remaining `prompt_len - matched` tokens.

The first cut only did the whole-entry case, and it sailed through every unit
test — then exp2's shared-system-prompt workload reported a flat
`prefill_skip_ratio = 0.000`, because `system + turn_i` is never a whole prefix
of `system + turn_j` (Pitfall 33). The common-run version is what makes the
realistic case work.

Why a *common leading run* and not RAG-style middle-segment splicing: with
causal attention and RoPE, a leading run's K/V depends only on the tokens up to
that point and their absolute positions — **not on what follows**. So a prefix
cloned from an earlier request (or that clone cropped shorter) is bit-identical
to prefilling it from scratch, *provided* the offset is the same and the
position ids are contiguous. Splice a shared segment in at a different offset
and that guarantee is gone. This track only does the safe case.

The clone itself is `copy.deepcopy` (`clone_dynamic_cache`). That's a real
dependency on deepcopy faithfully reproducing every per-layer key/value tensor
(Pitfall 35); `tests/test_kv_cache_manager.py::test_clone_is_faithful` pins it
by evolving an original and a clone through identical `update()` calls and
asserting the layer tensors stay bit-identical.

The speculative path has **two** caches per sequence, draft and target, and
each stores its own prefix — `seed` clones both, `put` stores both. They are
never assumed interchangeable (Pitfall B-10 in the plan).

## Experiment 1 — capacity admission vs a contiguous baseline

`src/verify_p7_2_paging.py`, real Qwen pair on MPS. Workload: 32 prompts with
lengths pulled apart by a filler segment (`random.choice([0,5,15,30])`,
`random.seed(0)`). For `total_blocks in {64, 128, 256, 512}`:

- **paged**: `kv_total_blocks = total_blocks`, `max_active = 64`. Admission is
  capacity-gated; record the peak concurrency actually reached.
- **contiguous baseline**: model the worst case of contiguous allocation —
  every slot must reserve room for the *longest* prompt — as
  `max_active = total_blocks // blocks_for(max_prompt_len + max_new_tokens)`,
  `kv_total_blocks = 0`.

The worst-case prompt needs 35 blocks (560 tokens), so the contiguous baseline
reserves that per slot.

| blocks | paged concurrency | contiguous concurrency | ratio | paged tok/s | contiguous tok/s | MemoryError |
|--:|--:|--:|--:|--:|--:|--:|
| 64  | 6  | 1  | 6.0 | 19.8 | 19.8 | 0 |
| 128 | 11 | 3  | 3.7 | 20.3 | 20.3 | 0 |
| 256 | 17 | 7  | 2.4 | 19.1 | 18.9 | 0 |
| 512 | 29 | 14 | 2.1 | 19.0 | 20.4 | 0 |

At the tightest budget (64 blocks) capacity-gated admission runs **6x** the
concurrency of the worst-case contiguous reservation, and no budget ever raises
`MemoryError` — admission defers, it never over-commits. Aggregate tok/s is
essentially flat across the pair (per-sequence caches, note 03: batch size
barely moves throughput on this stack), so the deliverable here is *how many
sequences you can safely admit for a given KV budget*, not tokens per second.
The ratio shrinks toward 2x as the budget grows because the fixed
worst-case-per-slot penalty of the contiguous model is a smaller fraction of a
large pool.

## Experiment 2 — shared system-prompt prefix reuse

One ~300-token filler "system prompt" prepended to 16 different user turns
(`SEGMENT_A + SEGMENT_B`). `prefix_cache in {False, True}`,
`kv_total_blocks = 512`, `max_active = 8`. Metric: `prefill_skip_ratio =
prefill_tokens_skipped / prefill_tokens_total`.

| prefix_cache | prefill_skip_ratio | agg tok/s | mean queue wait (rounds) |
|:--|--:|--:|--:|
| off | 0.00 | 18.1 | 7.4 |
| on  | 0.87 | 20.5 | 7.7 |

With the store on, 87% of prefill-forward positions across the 16 requests are
served from the cloned-and-cropped system-prompt KV instead of recomputed. That
buys back ~13% aggregate tok/s here (the ~300 shared tokens are a large slice of
each ~380-token prompt). Queue wait barely moves — this workload is
throughput-bound, not admission-bound, at `kv_total_blocks = 512`. The first
request pays full prefill and seeds the entry; the other 15 hit the common run
(none is a whole prefix of another, so this is the Pitfall 33 path).

## Oracle coverage

- **`run_o1_prefix`** (`src/spec_oracles.py`, wired into
  `tests/test_spec_faultlib.py`): two prompts sharing a leading run; one
  decoded from scratch with `speculative_generate_kv`, one seeded from a mini
  `PrefixStore` clone and driven from there. Output tokens must match exactly.
  This is the property that makes prefix reuse safe to ship, stated as a test.
- **Pool accounting invariant** (`test_serving_loop_paging.py`): step a
  small-pool server that forces queueing and assert `pool.free_blocks() >= 0`
  and `sum(used) + shared <= total` after every round.
- **Output identity**: a pool+prefix server and an all-off server produce
  byte-identical token streams on the FakeModel pair.

## Honest boundaries

- **No kernel-level batch speedup** — same as note 03. This changes *which*
  requests are admitted and *how much* prefill runs, not how fast a forward
  pass is.
- **Reuse is common-leading-run only.** A shared prefix followed by divergent
  tails works (the store crops the clone); a shared *middle* segment does not —
  that needs position-id surgery this track doesn't do. A partial match also
  doesn't re-store the longer prompt (`put` only on a miss), so a later
  identical tail won't get a longer hit.
- **Pool accounting is prompt-footprint only.** `acquire` reserves blocks for
  the prompt; the sequence's own generated tokens extend the real cache
  without a per-token `grow` call. On a 48-new-token cap that's a small
  under-count; a production version would call `grow` each round.
- **`block_size = 16` and the contiguous-baseline model are stylized.** The
  "reserve the longest prompt per slot" baseline is the worst case for
  contiguous allocation, not a measurement of a real contiguous allocator.
