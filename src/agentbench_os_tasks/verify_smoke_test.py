"""
Verifier self-check for every task in registry.py -- run this, not manual
inspection, to confirm a verify() function actually discriminates instead
of silently always passing (notes/project_plan_v9.md §9.6 风险3 applied
to P3.0's verifiers instead of P1.2's greedy comparator).

For every task, including held-out ones:
  1. setup() alone (untouched initial state) -> verify() must FAIL.
     A verifier that passes on the untouched fixture isn't testing anything.
  2. setup() then golden_solution() -> verify() must PASS.
     Confirms the correct answer is actually accepted.

Running golden_solution() here is not an evaluation run in the 9.6风险1
sense (see registry.py's module docstring) -- it never touches a model or
measures acceptance rate, so doing this for held-out tasks does not violate
"only run the held-out set once, after hyperparameters are finalized."

Usage: `python -m agentbench_os_tasks.verify_smoke_test` from src/, or
`python -m src.agentbench_os_tasks.verify_smoke_test` from the repo root.
No LLM calls, no network access beyond what Python's stdlib needs.
"""
from __future__ import annotations

import sys

from .registry import ALL_TASKS
from .sandbox import fresh_workdir


def run_smoke_tests() -> list[tuple[str, bool, str]]:
    """Returns (task_id, ok, detail) for every task. ok=True means both the
    untouched-state-fails and golden-solution-passes checks succeeded."""
    results: list[tuple[str, bool, str]] = []
    for task in ALL_TASKS:
        with fresh_workdir(task) as workdir:
            untouched = task.verify(workdir)
        if untouched.passed:
            results.append((task.task_id, False, f"verify() passed on UNTOUCHED input: {untouched.detail}"))
            continue

        with fresh_workdir(task) as workdir:
            task.golden_solution(workdir)
            golden = task.verify(workdir)
        if not golden.passed:
            results.append((task.task_id, False, f"verify() FAILED on golden_solution() output: {golden.detail}"))
            continue

        results.append((task.task_id, True, "untouched correctly rejected, golden correctly accepted"))
    return results


def main() -> int:
    results = run_smoke_tests()
    failures = [r for r in results if not r[1]]
    for task_id, ok, detail in results:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {task_id}: {detail}")
    print(f"\n{len(results) - len(failures)}/{len(results)} verifiers passed self-check")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
