"""
P7 -- goodput model for adaptive speculation length.

Reproduces the SmartSpec / TurboSpec estimator (arXiv:2406.14066):
choose per-round speculation length k in [k_min, k_max] to maximise

    goodput(k) = E[accepted tokens this round] / E[round wall-time]

k = 0 means "do not speculate this round" (plain target decode).

Pure and hermetic: every timing/accept input is a float the caller supplies.
The linear round-time model's coefficients come from src/goodput_profile.py
(offline calibration on the real Qwen pair, written to
results/p7_0_goodput_profile.json).

Contrast with what this repo already has:
  * the alpha-floor circuit breaker (src/serving_loop.py) is a *binary* switch --
    speculate at a fixed gamma, or don't -- and it is batch-blind (坑15/20);
  * vLLM's dynamic speculative-decoding disables speculation by a running-average
    acceptance threshold bucketed by concurrency (a lookup table).
Here the decision is a continuous argmax over a goodput curve whose round-time
term explicitly grows with n_active, so k* contracts under load without a table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RoundTimeCoeffs:
    """Linear model of one serving round's wall-time.

        T_round ~= c0
                 + c1 * (n_active * (mean_pending + k))     # batched TARGET verify
                 + c2 * (n_active * mean_kv_len)             # KV read / attention
                 + c3 * (n_active * k)                       # DRAFT-side proposal cost

    c3 is the term the "dead zone" pitfall (坑14) is about: if the draft forward
    is not cheap relative to the target, speculation cannot pay, and a round-time
    model that omits the draft cost will always over-estimate the value of a
    large k.
    """

    c0: float          # fixed per-round seconds (kernel launch, python, sampling)
    c1: float          # seconds per token in the batched TARGET verify forward
    c2: float          # seconds per (active seq * KV context token)
    c3: float          # seconds per (active seq * speculative token) -- DRAFT-side cost
    r2: float = 0.0    # held-out fit quality, informational only
    n_fit: int = 0

    def as_dict(self) -> dict:
        return {
            "c0": self.c0,
            "c1": self.c1,
            "c2": self.c2,
            "c3": self.c3,
            "r2": self.r2,
            "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoundTimeCoeffs":
        return cls(
            c0=float(d["c0"]),
            c1=float(d["c1"]),
            c2=float(d["c2"]),
            c3=float(d["c3"]),
            r2=float(d.get("r2", 0.0)),
            n_fit=int(d.get("n_fit", 0)),
        )

    @classmethod
    def from_json(cls, path: str) -> "RoundTimeCoeffs":
        """Reads p7_0_goodput_profile.json["coeffs"]."""
        with open(path) as fh:
            blob = json.load(fh)
        return cls.from_dict(blob["coeffs"])


def expected_accepted_tokens(alpha: float, k: int) -> float:
    """E[accepted drafts + 1 bonus] for ONE sequence (Leviathan et al. 2023).

        (1 - alpha**(k+1)) / (1 - alpha)

    k <= 0 -> 1.0 (bonus only, no speculation). alpha -> 1 -> limit is k + 1
    (坑30: the closed form divides by 1 - alpha, so the limit is taken
    explicitly rather than left to a 0/0).
    """
    if k <= 0:
        return 1.0
    if alpha >= 1.0 - 1e-9:
        return float(k + 1)
    if alpha <= 1e-12:
        return 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def expected_round_time(coeffs: RoundTimeCoeffs, *, n_active: int,
                        mean_pending: float, mean_kv_len: float, k: int) -> float:
    verify_tokens = n_active * (mean_pending + k)
    return (coeffs.c0
            + coeffs.c1 * verify_tokens
            + coeffs.c2 * n_active * mean_kv_len
            + coeffs.c3 * n_active * k)


def goodput(coeffs: RoundTimeCoeffs, *, alpha: float, n_active: int,
            mean_pending: float, mean_kv_len: float, k: int) -> float:
    acc = n_active * expected_accepted_tokens(alpha, k)
    t = expected_round_time(coeffs, n_active=n_active, mean_pending=mean_pending,
                            mean_kv_len=mean_kv_len, k=k)
    return acc / t if t > 0 else 0.0


def best_k(coeffs: RoundTimeCoeffs, *, alpha: float, n_active: int,
           mean_pending: float, mean_kv_len: float,
           k_min: int = 0, k_max: int = 8,
           prev_k: int | None = None, hysteresis: int = 0):
    """Returns (k_star, {k: goodput(k)}).

    If prev_k is given and hysteresis > 0, k_star is clamped to
    prev_k +/- hysteresis so the controller cannot swing from 8 to 0 in one
    round (坑31 -- a per-round argmax over a noisy alpha estimate chatters).
    """
    scores = {k: goodput(coeffs, alpha=alpha, n_active=n_active,
                         mean_pending=mean_pending, mean_kv_len=mean_kv_len, k=k)
              for k in range(k_min, k_max + 1)}
    k_star = max(scores, key=lambda kk: scores[kk])
    if prev_k is not None and hysteresis > 0:
        lo, hi = max(k_min, prev_k - hysteresis), min(k_max, prev_k + hysteresis)
        k_star = min(hi, max(lo, k_star))
    return k_star, scores
