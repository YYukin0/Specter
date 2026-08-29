"""~95%-reproduction gate for Bullet 2 (支柱7 TASKS.md), per
notes/简历定稿计划-Specter_2026-08-28.md: "先跑 ~95% 论文复现自检确认没配错" --
before spending the rest of the ~$15 budget on the full arm x concurrency
matrix, confirm the eagle3 arm at concurrency=1 lands within 95% of the
published EAGLE3-8B speedup band (1.25-1.32x). This catches a misconfigured
--speculative-config, a mismatched draft/target tokenizer, or a broken
warm-up before those mistakes get baked into every subsequent measurement.

Run after a single eagle3-vs-baseline pair at concurrency=1
(see notes/cloud-bullet2-execution-plan_2026-08-29.md step 4):

    python -m cloud_bench.sanity_check \
        --eagle3-tok-per-s <measured> --baseline-tok-per-s <measured>
"""
from __future__ import annotations

import argparse
import sys

from cloud_bench import config


def check_reproduction(measured_speedup: float) -> tuple[bool, str]:
    threshold = config.PAPER_SPEEDUP_LOW * config.SANITY_THRESHOLD_FRACTION
    ok = measured_speedup >= threshold
    verdict = "PASS" if ok else "FAIL -- stop, do not run the full matrix yet"
    msg = (
        f"measured c=1 eagle3 speedup {measured_speedup:.3f}x vs paper band "
        f"[{config.PAPER_SPEEDUP_LOW}, {config.PAPER_SPEEDUP_HIGH}]x, "
        f"threshold {threshold:.3f}x ({config.SANITY_THRESHOLD_FRACTION:.0%} of low end) -> {verdict}"
    )
    return ok, msg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eagle3-tok-per-s", type=float, required=True)
    p.add_argument("--baseline-tok-per-s", type=float, required=True)
    args = p.parse_args(argv)

    if args.baseline_tok_per_s <= 0:
        print("baseline throughput is zero or missing -- check the server actually ran "
              "before trusting this number", file=sys.stderr)
        return 2

    speedup = args.eagle3_tok_per_s / args.baseline_tok_per_s
    ok, msg = check_reproduction(speedup)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
