"""
P6.3 -- terminal live view for the speculative-decoding serving loop.

Runs `SpecServer` (src/serving_loop.py) over a fixed prompt set and redraws a
plain-ANSI dashboard after every round: streaming text per active request, and
per-round gamma / accept length / rolling alpha / tok-s / concurrency / breaker
state. Three toggles show the speed difference live:

    --no-spec       speculation OFF   -> every round is plain target decoding
    --gammatune     adaptive gamma ON -> gamma moves with the batch accept rate
    --no-breaker    circuit breaker OFF

stdlib + ANSI only, no new dependencies. `--fake` swaps in the deterministic
FakeModel pair so the demo (and its test) runs with no model download.

Examples:
    python -m demo.live                       # all defaults, real Qwen pair
    python -m demo.live --no-spec              # watch it crawl
    python -m demo.live --gammatune --compare  # A/B a config against baseline
    python -m demo.live --fake --no-live       # headless, for CI
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving_loop import ServeConfig, SpecServer  # noqa: E402

DEMO_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a short haiku about autumn.",
    "What are three tips for staying focused while studying?",
    "Summarize the plot of Cinderella in one paragraph.",
    "Give me a recipe idea for a quick weeknight dinner.",
    "Write a Python function that returns the nth Fibonacci number.",
]

# ----------------------------------------------------------------------------- #
# ANSI helpers (no curses, no rich)
# ----------------------------------------------------------------------------- #
CSI = "\x1b["
CLEAR = CSI + "2J"
HOME = CSI + "H"
HIDE_CUR = CSI + "?25l"
SHOW_CUR = CSI + "?25h"


def _c(s: str, code: str) -> str:
    return f"{CSI}{code}m{s}{CSI}0m"


def dim(s):    return _c(s, "2")
def bold(s):   return _c(s, "1")
def green(s):  return _c(s, "32")
def yellow(s): return _c(s, "33")
def red(s):    return _c(s, "31")
def cyan(s):   return _c(s, "36")


def _onoff(flag: bool) -> str:
    return green(" ON") if flag else dim("OFF")


def _mode_tag(mode: str) -> str:
    return {"spec": green("spec"), "degraded": red("degraded"),
            "probe": yellow("probe"), "idle": dim("idle")}.get(mode, mode)


# ----------------------------------------------------------------------------- #
@dataclass
class DemoResult:
    label: str
    wall_s: float
    total_tokens: int
    agg_tok_per_s: float
    n_rounds: int
    mode_counts: Dict[str, int]
    mean_accept_len: float
    rolling_alpha_end: float
    gamma_start: int
    gamma_end: int
    realignment_overhead_mean: float
    texts: Dict[str, str] = field(default_factory=dict)

    def summary_line(self) -> str:
        mc = " ".join(f"{k}:{v}" for k, v in self.mode_counts.items())
        return (f"{self.label:<22} {self.agg_tok_per_s:6.1f} tok/s  "
                f"{self.total_tokens:4d} tok / {self.wall_s:5.1f}s  "
                f"rounds={self.n_rounds:3d} [{mc}]  "
                f"accept_len={self.mean_accept_len:.2f}  "
                f"gamma {self.gamma_start}->{self.gamma_end}")


# ----------------------------------------------------------------------------- #
def _load_pair(fake: bool):
    if fake:
        from spec_oracles import LengthOnlyCache, make_fake_pair
        draft, target, tok = make_fake_pair(phase_target=0.3)
        return draft, target, tok, LengthOnlyCache
    from model_loader import (DRAFT_MODEL_NAME, TARGET_MODEL_NAME,
                              load_model_and_tokenizer)
    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)
    from spec_kv import _new_cache
    return draft, target, tok, _new_cache


def _tail(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else "..." + s[-(n - 3):]


def _render(server: SpecServer, cfg: ServeConfig, label: str,
            t0: float, emitted_so_far: int, last: "RoundInfoLike") -> str:
    el = time.perf_counter() - t0
    lines = [
        bold(f"  speculative-decoding serving loop  --  {label}"),
        dim(f"  draft/target: {getattr(server, '_names', 'Qwen2.5 0.5B / 1.5B')}"),
        "",
        (f"  speculation {_onoff(cfg.spec_enabled)}    "
         f"adaptive-gamma {_onoff(cfg.gammatune_on)}    "
         f"circuit-breaker {_onoff(cfg.breaker_on)}"),
        "",
        (f"  round {last.index:<4d} mode {_mode_tag(last.mode):<20} "
         f"gamma {last.round_gamma:<2d}  active {last.n_active}  queued {last.n_queued}"),
        (f"  rolling alpha {last.rolling_alpha:4.2f}   "
         f"realign-tax {last.realignment_overhead:4.2f}   "
         f"agg {emitted_so_far / el if el else 0:6.1f} tok/s   "
         f"elapsed {el:5.1f}s"),
        dim(f"  breaker: {last.breaker_reason}"),
        "",
        dim("  " + "-" * 74),
    ]
    for s in server.active:
        txt = server.tok.decode(s.token_ids, skip_special_tokens=True) if s.token_ids else ""
        al = s.accept_lengths[-1] if s.accept_lengths else 0
        lines.append(f"  {cyan(s.req_id):<14} a={s.alpha:4.2f} +{al}  "
                     f"{_tail(txt, 52)}")
    done = server.results()
    if done:
        lines.append("")
        lines.append(dim(f"  finished {len(done)}/{len(DEMO_PROMPTS)}:"))
        for rid, r in list(done.items())[-3:]:
            lines.append(dim(f"  {rid:<14} {_tail(r.text, 58)}"))
    return "\n".join(lines)


class RoundInfoLike:
    """Minimal stand-in used before the first real round is logged."""
    index = 0
    mode = "idle"
    round_gamma = 0
    n_active = 0
    n_queued = 0
    rolling_alpha = 1.0
    realignment_overhead = 0.0
    breaker_reason = "starting"


# ----------------------------------------------------------------------------- #
def run_demo(*, spec: bool = True, gammatune: bool = False, breaker: bool = True,
             fake: bool = False, live: bool = True, max_active: int = 3,
             gamma: int = 4, max_new_tokens: int = 48, n_prompts: int = 6,
             frame_min_s: float = 0.0, out=sys.stdout, label: Optional[str] = None,
             pair=None) -> DemoResult:
    draft, target, tok, make_cache = pair or _load_pair(fake)
    cfg = ServeConfig(
        gamma=gamma, temperature=1.0, max_new_tokens=max_new_tokens,
        max_active=max_active, make_cache=make_cache,
        spec_enabled=spec, gammatune_on=gammatune, breaker_on=breaker,
        alpha_floor=0.5, warmup_rounds=3, reprobe_every=12,
        apply_chat_template=True,   # FakeModel _Tok supports apply_chat_template
    )
    server = SpecServer(draft, target, tok, cfg)
    server._names = "FakeModel pair" if fake else "Qwen2.5 0.5B / 1.5B"
    prompts = DEMO_PROMPTS[:n_prompts]
    for i, p in enumerate(prompts):
        server.submit(p, req_id=f"req{i}", seed=1000 + i)

    label = label or _auto_label(spec, gammatune, breaker)
    if live:
        out.write(HIDE_CUR + CLEAR)
    t0 = time.perf_counter()
    emitted = 0
    last = RoundInfoLike()
    modes: Dict[str, int] = {}
    reali: List[float] = []

    while server.active or server.pending:
        fr = time.perf_counter()
        last = server.step()
        emitted += last.emitted
        modes[last.mode] = modes.get(last.mode, 0) + 1
        if last.mode != "idle":
            reali.append(last.realignment_overhead)
        if live:
            out.write(HOME + _render(server, cfg, label, t0, emitted, last) + "\n")
            out.flush()
            slack = frame_min_s - (time.perf_counter() - fr)
            if slack > 0:
                time.sleep(slack)

    wall = time.perf_counter() - t0
    if live:
        out.write(SHOW_CUR)

    res = server.results()
    total = sum(len(r.token_ids) for r in res.values())
    accs = [a for r in res.values() for a in r.accept_lengths if r.accept_lengths]
    return DemoResult(
        label=label, wall_s=wall, total_tokens=total,
        agg_tok_per_s=total / wall if wall else 0.0,
        n_rounds=sum(modes.values()),
        mode_counts=modes,
        mean_accept_len=sum(accs) / len(accs) if accs else 0.0,
        rolling_alpha_end=last.rolling_alpha,
        gamma_start=gamma, gamma_end=cfg.gamma,
        realignment_overhead_mean=sum(reali) / len(reali) if reali else 0.0,
        texts={rid: r.text for rid, r in res.items()},
    )


def _auto_label(spec, gammatune, breaker) -> str:
    if not spec:
        return "speculation OFF"
    bits = ["spec"]
    if gammatune:
        bits.append("gammatune")
    bits.append("breaker" if breaker else "no-breaker")
    return "+".join(bits)


# ----------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-spec", action="store_true", help="speculation off")
    ap.add_argument("--gammatune", action="store_true", help="adaptive gamma on")
    ap.add_argument("--no-breaker", action="store_true", help="circuit breaker off")
    ap.add_argument("--fake", action="store_true", help="deterministic FakeModel, no download")
    ap.add_argument("--no-live", action="store_true", help="no ANSI redraw (headless)")
    ap.add_argument("--compare", action="store_true",
                    help="also run the speculation-off baseline and print the speedup")
    ap.add_argument("--max-active", type=int, default=3)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--fps", type=float, default=8.0, help="max redraw rate")
    args = ap.parse_args(argv)

    live = not args.no_live and sys.stdout.isatty()
    pair = _load_pair(args.fake)   # load once, reuse for --compare
    common = dict(fake=args.fake, live=live, max_active=args.max_active,
                  gamma=args.gamma, max_new_tokens=args.max_new_tokens,
                  n_prompts=args.prompts, pair=pair,
                  frame_min_s=(1.0 / args.fps if args.fps else 0.0))

    main_run = run_demo(spec=not args.no_spec, gammatune=args.gammatune,
                        breaker=not args.no_breaker, **common)
    print("\n" + bold("result"))
    print("  " + main_run.summary_line())

    if args.compare:
        base = run_demo(spec=False, gammatune=False, breaker=False,
                        label="speculation OFF", **common)
        print("  " + base.summary_line())
        if base.agg_tok_per_s:
            print(bold(f"\n  speedup vs speculation-off: "
                       f"{main_run.agg_tok_per_s / base.agg_tok_per_s:.2f}x"))


if __name__ == "__main__":
    main()
