"""
P6.1 -- `SpecServer`: a single-threaded, deterministic serving loop with
continuous batching on top of the per-sequence-cache batched speculative decoder
(src/spec_kv_batch.py).

  * `submit(prompt) -> req_id`   queue a request
  * `step() -> RoundInfo`        advance every active sequence by one round;
                                admit queued requests into freed slots first
  * `poll(req_id) -> ReqResult | None`
  * `run_until_idle()`           step() until nothing is active or queued

Continuous batching: the moment a sequence finishes, its slot is refilled from
the queue on the next `step()` -- the loop never drains the whole batch to start
new work.

Circuit breaker on REAL signals (deployment-depth-plan sec 7 C4). The P5.3
breaker tripped on a synthetic `batch_size >= threshold` trace (sec 9.2 坑15:
that signal never fed back into alpha, so "speculation gets worse at high batch"
never actually happened in the numbers). Here the breaker watches:

  * rolling acceptance rate `alpha` over the last `window` accept decisions
    across all sequences -- the quantity that actually determines whether
    speculation pays (MLX guidance: net win needs alpha > ~0.65; the literature
    net-loss line is alpha < ~0.5);
  * a periodic target-only latency probe vs the measured speculative round
    latency -- a gamma-tolerance-style check that the speculative round is not
    actually slower wall-clock (the C4 "rethink big-batch" criterion);
  * `len(active)` is only an *input* to those, never the rule itself.

Trip: rolling alpha < `alpha_floor` (after a warmup) OR the last speculative
round was slower than the target-only latency estimate for the same tokens.
While degraded, force one speculative probe every `reprobe_every` rounds so the
breaker can notice alpha recovering (P5.3 坑11: a breaker that never re-measures
can never re-enable).
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import torch

from gammatune import gammatune_update
from rejection_sampling import collect_eos_ids
from spec_kv_batch import RoundTelemetry, SeqState, make_seq, run_round
from spec_kv import _cache_position, _new_cache


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class ServeConfig:
    gamma: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 64
    max_active: int = 8
    apply_chat_template: bool = True
    seed_base: int = 0
    make_cache: object = _new_cache      # injectable for hermetic FakeModel tests
    # demo toggles
    spec_enabled: bool = True           # False -> every round is plain target decoding
    gammatune_on: bool = False          # True -> adapt cfg.gamma per round (batch-mean A)
    gamma_min: int = 1
    gamma_max: int = 10
    # circuit breaker
    breaker_on: bool = True
    alpha_floor: float = 0.5            # rolling alpha below this -> degrade
    alpha_window: int = 64             # accept decisions kept for the rolling mean
    warmup_rounds: int = 4            # no trip before this many speculative rounds
    reprobe_every: int = 20            # while degraded, force a spec probe this often
    latency_probe_every: int = 0      # >0: run a target-only latency probe every N rounds
    latency_slack: float = 1.10       # spec round may be up to this x the target-only estimate
    # P7 Track C -- adaptive speculation-length controller (default = old behavior)
    controller: str = "alpha_floor"    # "alpha_floor" | "goodput" | "fixed"
    goodput_coeffs_path: str = "results/p7_0_goodput_profile.json"
    goodput_coeffs: object = None      # injectable RoundTimeCoeffs for hermetic tests; overrides path
    k_min: int = 0
    k_max: int = 8
    controller_hysteresis: int = 2
    controller_warmup: int = 4
    # P7 Track B -- block-structured KV accounting + exact-prefix reuse (default off)
    kv_total_blocks: int = 0           # 0 -> pool disabled (current behavior)
    kv_block_size: int = 16
    prefix_cache: bool = False         # requires kv_total_blocks > 0
    prefix_cache_max_entries: int = 32


@dataclass
class ReqResult:
    req_id: str
    prompt: str
    text: str
    token_ids: List[int]
    n_rounds: int
    alpha: float
    accept_lengths: List[int]
    prompt_len: int
    eos_hit: bool
    submit_round: int
    admit_round: int
    finish_round: int

    @property
    def queue_wait_rounds(self) -> int:
        return max(0, self.admit_round - self.submit_round)


@dataclass
class RoundInfo:
    index: int
    mode: str                          # "spec" | "degraded" | "probe"
    n_active: int
    n_queued: int
    admitted: List[str]
    finished: List[str]
    rolling_alpha: float
    realignment_overhead: float
    emitted: int
    wall_s: float
    breaker_reason: str
    round_gamma: int = 0               # gamma used this round (moves when gammatune_on)
    controller_k: int = -1             # goodput controller's k* this round (-1 = not the goodput controller)


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
class SpecServer:
    def __init__(self, draft_model, target_model, tokenizer, config: Optional[ServeConfig] = None):
        self.draft = draft_model
        self.target = target_model
        self.tok = tokenizer
        self.cfg = config or ServeConfig()
        self.device = next(target_model.parameters()).device
        # input_ids dtype -- speculative_step_kv builds `torch.tensor([...], dtype=)`
        # for token ids, so this is long, NOT the model weight dtype.
        self.dtype = torch.long
        self.eos_ids = collect_eos_ids(tokenizer, target_model)

        self.pending: Deque[Tuple[str, str, int, Optional[int]]] = collections.deque()
        self.active: List[SeqState] = []
        self.finished: Dict[str, ReqResult] = {}
        self.round_index = 0
        self._auto_id = 0

        # breaker state
        self._alpha_hist: Deque[Tuple[int, int]] = collections.deque()  # (accepted, evaluated)
        self._alpha_acc = 0
        self._alpha_ev = 0
        self.mode = "spec"                       # underlying mode: "spec" | "degraded"
        self._spec_rounds_seen = 0
        self._rounds_since_reprobe = 0
        self._last_target_only_ms: Optional[float] = None
        self._gamma_bar = float(self.cfg.gamma)   # GammaTune EMA (demo toggle)

        self.round_log: List[RoundInfo] = []
        self.telemetry: List[RoundTelemetry] = []

        # P7 Track C -- goodput controller state
        self._controller_coeffs = None
        self._last_k = self.cfg.gamma
        if self.cfg.controller == "goodput":
            from goodput_model import RoundTimeCoeffs
            self._controller_coeffs = (
                self.cfg.goodput_coeffs
                or RoundTimeCoeffs.from_json(self.cfg.goodput_coeffs_path)
            )

        # P7 Track B -- block KV pool + exact-prefix store (both default-disabled)
        self.pool = None
        self.prefix_store = None
        if self.cfg.kv_total_blocks > 0:
            from kv_cache_manager import BlockKVPool, PrefixStore
            self.pool = BlockKVPool(self.cfg.kv_total_blocks, self.cfg.kv_block_size)
            if self.cfg.prefix_cache:
                self.prefix_store = PrefixStore(self.pool, self.cfg.prefix_cache_max_entries)
        self.prefill_tokens_total = 0
        self.prefill_tokens_skipped = 0

    # ------------------------------------------------------------------ #
    def submit(self, prompt: str, *, req_id: Optional[str] = None,
               seed: Optional[int] = None, max_new_tokens: Optional[int] = None) -> str:
        if req_id is None:
            req_id = f"r{self._auto_id}"
            self._auto_id += 1
        s = seed if seed is not None else self.cfg.seed_base + self._hash_seed(req_id)
        mnt = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        self.pending.append((req_id, prompt, mnt, s))
        return req_id

    @staticmethod
    def _hash_seed(req_id: str) -> int:
        return int.from_bytes(req_id.encode()[:4].ljust(4, b"\0"), "little") % 100_000

    # ------------------------------------------------------------------ #
    def _admit(self) -> List[str]:
        admitted: List[str] = []
        if self.pool is None and self.prefix_store is None:
            # unchanged pre-P7 path
            while self.pending and len(self.active) < self.cfg.max_active:
                req_id, prompt, mnt, seed = self.pending.popleft()
                seq = make_seq(req_id, prompt, self.tok, device=self.device,
                               max_new_tokens=mnt, seed=seed,
                               apply_chat_template=self.cfg.apply_chat_template,
                               make_cache=self.cfg.make_cache)
                seq.submit_round = getattr(seq, "submit_round", 0)
                seq.admit_round = self.round_index
                self.active.append(seq)
                admitted.append(req_id)
            return admitted

        # P7 Track B: capacity-driven admission + exact-prefix reuse
        while self.pending and len(self.active) < self.cfg.max_active:
            req_id, prompt, mnt, seed = self.pending[0]
            seq = make_seq(req_id, prompt, self.tok, device=self.device,
                           max_new_tokens=mnt, seed=seed,
                           apply_chat_template=self.cfg.apply_chat_template,
                           make_cache=self.cfg.make_cache)
            prompt_len = len(seq.committed)
            matched, entry = (0, None)
            if self.prefix_store is not None:
                matched, entry = self.prefix_store.longest_match(seq.committed)
            need = prompt_len - matched
            if self.pool is not None and not self.pool.can_admit(need):
                break                              # leave queued; retry next step

            self.pending.popleft()
            if entry is not None and self.prefix_store is not None:
                d, t, ds, ts = self.prefix_store.seed(entry)
                seq.draft_cache, seq.target_cache = d, t
                seq.draft_synced, seq.target_synced = ds, ts
                self.prefill_tokens_skipped += matched
            if self.pool is not None:
                self.pool.acquire(req_id, need)
            if self.prefix_store is not None:
                self._prefill_seq(seq, prompt_len)
                if entry is None:
                    self.prefix_store.put(list(seq.committed[:prompt_len]),
                                          seq.draft_cache, seq.target_cache)
            self.prefill_tokens_total += prompt_len
            seq.submit_round = getattr(seq, "submit_round", 0)
            seq.admit_round = self.round_index
            self.active.append(seq)
            admitted.append(req_id)
        return admitted

    @property
    def prefill_skip_ratio(self) -> float:
        return self.prefill_tokens_skipped / max(1, self.prefill_tokens_total)

    @torch.no_grad()
    def _prefill_seq(self, seq, upto: int) -> None:
        """Bring `seq`'s draft/target caches to length `upto - 1` (rectangular
        invariant -- the last prompt token stays as round 1's pending input).
        One forward each over the not-yet-cached slice."""
        lo = seq.target_synced
        hi = upto - 1
        if hi <= lo:
            seq.draft_synced = seq.target_synced = max(lo, hi)
            return
        chunk = seq.committed[lo:hi]
        feed = torch.tensor([chunk], device=self.device, dtype=self.dtype)
        pos = _cache_position(lo, len(chunk), self.device)
        self.draft(input_ids=feed, past_key_values=seq.draft_cache,
                   use_cache=True, cache_position=pos)
        self.target(input_ids=feed, past_key_values=seq.target_cache,
                    use_cache=True, cache_position=pos)
        seq.draft_synced = hi
        seq.target_synced = hi

    # ------------------------------------------------------------------ #
    def _rolling_alpha(self) -> float:
        return self._alpha_acc / self._alpha_ev if self._alpha_ev else 1.0

    def _push_alpha(self, accepted: int, evaluated: int) -> None:
        if evaluated <= 0:
            return
        self._alpha_hist.append((accepted, evaluated))
        self._alpha_acc += accepted
        self._alpha_ev += evaluated
        while self._alpha_ev > self.cfg.alpha_window and len(self._alpha_hist) > 1:
            a, e = self._alpha_hist.popleft()
            self._alpha_acc -= a
            self._alpha_ev -= e

    def _decide_mode(self) -> Tuple[str, str]:
        """Return (round_mode, reason). round_mode in {"spec","degraded","probe"}."""
        cfg = self.cfg
        if not cfg.spec_enabled:
            return "degraded", "speculation disabled (demo toggle)"
        if not cfg.breaker_on or cfg.controller == "fixed":
            return "spec", "fixed / breaker off"

        alpha = self._rolling_alpha()
        if self.mode == "spec":
            if self._spec_rounds_seen < cfg.warmup_rounds:
                return "spec", f"warmup ({self._spec_rounds_seen}/{cfg.warmup_rounds})"
            if alpha < cfg.alpha_floor:
                return "degraded", f"rolling alpha {alpha:.2f} < floor {cfg.alpha_floor}"
            if (self._last_target_only_ms is not None and self.telemetry
                    and self.telemetry[-1].mode == "spec"):
                spec_ms = self.telemetry[-1].wall_s * 1e3
                if spec_ms > cfg.latency_slack * self._last_target_only_ms:
                    return "degraded", (f"spec round {spec_ms:.0f}ms > "
                                        f"{cfg.latency_slack}x target-only {self._last_target_only_ms:.0f}ms")
            return "spec", f"rolling alpha {alpha:.2f} ok"

        # degraded
        if self._rounds_since_reprobe >= cfg.reprobe_every:
            return "probe", "periodic probe while degraded (坑11)"
        if alpha >= cfg.alpha_floor and self._spec_rounds_seen > 0:
            # a prior probe lifted alpha back over the floor -> restore
            return "spec", f"rolling alpha {alpha:.2f} recovered >= floor"
        return "degraded", f"rolling alpha {alpha:.2f} still < floor {cfg.alpha_floor}"

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _latency_probe(self) -> None:
        """Time one target-only forward over an active sequence's committed
        prefix -- the wall-clock a plain autoregressive step would cost. Cheap,
        and it is the C4 gamma-tolerance reference."""
        if not self.active:
            return
        s = self.active[0]
        feed = torch.tensor([s.committed[-1:]], device=self.device, dtype=self.dtype)
        pos = _cache_position(max(0, len(s.committed) - 1), 1, self.device)
        t0 = time.perf_counter()
        self.target(input_ids=feed, cache_position=pos)
        self._last_target_only_ms = (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------ #
    def step(self) -> RoundInfo:
        admitted = self._admit()
        n_queued = len(self.pending)

        if not self.active:
            info = RoundInfo(self.round_index, "idle", 0, n_queued, admitted, [],
                             self._rolling_alpha(), 0.0, 0, 0.0, "no active sequences",
                             round_gamma=self.cfg.gamma)
            self.round_log.append(info)
            self.round_index += 1
            return info

        if self.cfg.latency_probe_every and self.round_index % self.cfg.latency_probe_every == 0:
            self._latency_probe()

        if (self.cfg.controller == "goodput"
                and self._spec_rounds_seen >= self.cfg.controller_warmup
                and self.active):
            from goodput_model import best_k
            alpha = self._rolling_alpha()
            n = len(self.active)
            mean_pending = sum(len(s.committed) - s.target_synced for s in self.active) / n
            mean_kv = sum(s.target_synced for s in self.active) / n
            k_star, _ = best_k(
                self._controller_coeffs, alpha=alpha, n_active=n,
                mean_pending=mean_pending, mean_kv_len=mean_kv,
                k_min=self.cfg.k_min, k_max=self.cfg.k_max,
                prev_k=self._last_k, hysteresis=self.cfg.controller_hysteresis,
            )
            self._last_k = k_star
            self.cfg.gamma = max(1, k_star)          # gamma>=1 for the actual round call
            if k_star == 0:
                round_mode, reason, exec_mode = (
                    "degraded", "goodput: k*=0 (spec not worth it)", "degraded")
                round_gamma = 1
            else:
                round_mode, reason, exec_mode = (
                    "spec", f"goodput: k*={k_star} (alpha={alpha:.2f}, n={n})", "spec")
                round_gamma = k_star
        else:
            round_mode, reason = self._decide_mode()
            exec_mode = "degraded" if round_mode == "degraded" else "spec"
            round_gamma = self.cfg.gamma

        tele = run_round(self.active, self.draft, self.target, gamma=round_gamma,
                         temperature=self.cfg.temperature, eos_ids=self.eos_ids,
                         device=self.device, dtype=self.dtype, mode=exec_mode)
        self.telemetry.append(tele)

        # breaker bookkeeping
        if exec_mode == "spec":
            self._spec_rounds_seen += 1
            for a, e in zip(tele.accepted_per_seq, tele.evaluated_per_seq):
                self._push_alpha(a, e)
            if self.cfg.gammatune_on and tele.accepted_per_seq:
                # demo-level adaptive gamma: one GammaTune update on the batch-mean
                # accepted length (per-stream control is P5.0; this is the toggle).
                a_mean = round(sum(tele.accepted_per_seq) / len(tele.accepted_per_seq))
                new_gamma, self._gamma_bar = gammatune_update(
                    round_gamma, self._gamma_bar, a_mean,
                    gmin=self.cfg.gamma_min, gmax=self.cfg.gamma_max)
                self.cfg.gamma = int(new_gamma)
        if round_mode == "spec":
            self.mode = "spec"
            self._rounds_since_reprobe = 0
        elif round_mode == "degraded":
            self.mode = "degraded"
            self._rounds_since_reprobe += 1
        elif round_mode == "probe":
            # ran a speculative round but stay degraded until alpha clears
            self._rounds_since_reprobe = 0

        finished = list(tele.finished_this_round)
        for s in list(self.active):
            if s.done:
                s.finish_round = self.round_index
                self.finished[s.req_id] = ReqResult(
                    req_id=s.req_id, prompt="", text=self.tok.decode(s.token_ids, skip_special_tokens=True),
                    token_ids=list(s.token_ids), n_rounds=s.n_rounds, alpha=s.alpha,
                    accept_lengths=list(s.accept_lengths), prompt_len=s.prompt_len,
                    eos_hit=s.eos_hit, submit_round=s.submit_round, admit_round=s.admit_round,
                    finish_round=self.round_index,
                )
                if self.pool is not None:
                    self.pool.release(s.req_id)
                self.active.remove(s)

        info = RoundInfo(
            index=self.round_index, mode=round_mode, n_active=tele.n_active,
            n_queued=n_queued, admitted=admitted, finished=finished,
            rolling_alpha=self._rolling_alpha(), realignment_overhead=tele.realignment_overhead,
            emitted=sum(tele.emitted_per_seq), wall_s=tele.wall_s, breaker_reason=reason,
            round_gamma=round_gamma,
            controller_k=(self._last_k if self.cfg.controller == "goodput" else -1),
        )
        self.round_log.append(info)
        self.round_index += 1
        return info

    # ------------------------------------------------------------------ #
    def poll(self, req_id: str) -> Optional[ReqResult]:
        return self.finished.get(req_id)

    def run_until_idle(self, max_rounds: int = 100_000) -> List[RoundInfo]:
        out: List[RoundInfo] = []
        while (self.active or self.pending) and len(out) < max_rounds:
            out.append(self.step())
        return out

    def results(self) -> Dict[str, ReqResult]:
        return dict(self.finished)
