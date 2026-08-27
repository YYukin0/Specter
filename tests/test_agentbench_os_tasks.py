"""Pytest wrapper around the P3.0 AgentBench-OS task package.

The package ships its own self-check (`agentbench_os_tasks.verify_smoke_test`)
that, for every task, asserts verify() rejects the untouched fixture and
accepts golden_solution()'s output -- the §9.6 风险3 fault-injection check
applied to P3.0's verifiers. This wrapper just makes `pytest tests/ -q` fail
if any of that regresses, plus pins the registry-shape invariants.

Framework files are `git show`-copied from origin/b/p3-0-agentbench-tasks
(reviewed, added as new files, not merged); the _05 tasks in file_ops /
code_refactor / cli_tools were added here to bring the active count to >= 12.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentbench_os_tasks.registry import (  # noqa: E402
    ACTIVE_TASKS,
    ALL_TASKS,
    HELD_OUT_TASKS,
)
from agentbench_os_tasks.schema import CATEGORIES  # noqa: E402
from agentbench_os_tasks.verify_smoke_test import run_smoke_tests  # noqa: E402


def test_every_verifier_discriminates():
    results = run_smoke_tests()
    failures = [(tid, detail) for tid, ok, detail in results if not ok]
    assert not failures, f"verifier self-check failures: {failures}"
    assert len(results) == len(ALL_TASKS)


def test_registry_shape():
    assert 15 <= len(ALL_TASKS) <= 20
    assert len(ACTIVE_TASKS) >= 12          # overnight queue #2 requirement
    assert 3 <= len(HELD_OUT_TASKS) <= 5
    assert {t.category for t in HELD_OUT_TASKS} == set(CATEGORIES)
    ids = [t.task_id for t in ALL_TASKS]
    assert len(ids) == len(set(ids))


def test_held_out_tasks_all_have_a_rationale():
    for t in HELD_OUT_TASKS:
        assert t.held_out_rationale, t.task_id


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
