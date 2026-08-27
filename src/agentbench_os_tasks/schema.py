"""
Task schema for the P3.0 AgentBench-OS-adjacent task set. See
notes/project_plan_v9.md §7 P3.0 + 附录B for the task-set spec this
implements, and TASKS.md M3 for where this sits in the two-dev plan.

Scope note: this module (and the rest of agentbench_os_tasks/) defines the
task *shape* and the *automatic verifiers* only. It does not run an agent
loop against any model -- that's P3.1 (blocked on P1, per TASKS.md M3),
which will import TaskSpec/ToolCallRecord/TaskRunResult from here to drive
a real model and populate a trace. Nothing in this package calls an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of running a task's verify() against a workdir. `passed` is
    the sole FINISHED signal -- no human judgment call anywhere in this
    package. `detail` is for debugging, not for deciding pass/fail."""

    passed: bool
    detail: str


@dataclass(frozen=True)
class TaskSpec:
    """One AgentBench-OS-adjacent task.

    setup(workdir), golden_solution(workdir), and verify(workdir) all
    operate on the same throwaway directory (see sandbox.py) and must not
    touch anything outside it.

    golden_solution exists purely to test the verifier itself (mirrors
    notes/project_plan_v9.md §9.6 风险3's fault-injection requirement for
    P1.2's correctness verifier): setup() then golden_solution() must make
    verify() pass, and setup() alone (untouched initial state) must make
    verify() fail. See verify_smoke_test.py, which asserts both directions
    for every task, including held-out ones -- running a golden solution
    through verify() is a test of the verifier's logic, not an evaluation
    run, so it does not fall under the held-out "run exactly once" rule in
    9.6 风险1 (that rule is about not tuning GammaTune's hyperparameters
    against these tasks; it has nothing to do with checking that the
    scoring script itself isn't broken).
    """

    task_id: str
    category: str
    description: str
    setup: Callable[[Path], None]
    golden_solution: Callable[[Path], None]
    verify: Callable[[Path], VerifyResult]
    held_out: bool = False
    held_out_rationale: str | None = None
    design_notes: str | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    """Shape stub for P3.1's engine trace. Nothing in P3.0 constructs a real
    one -- this only exists so P3.1 can import a stable interface instead of
    inventing its own ad hoc trace format later."""

    tool_name: str
    args: dict
    result_summary: str


@dataclass(frozen=True)
class TaskRunResult:
    """Shape stub for what P3.1 will produce by actually running a model
    against a TaskSpec. Not populated anywhere in P3.0."""

    task_id: str
    trace: list[ToolCallRecord]
    verify_result: VerifyResult
    wall_clock_s: float


CATEGORIES: tuple[str, ...] = (
    "file_ops",
    "code_refactor",
    "cli_tools",
    "multistep_dependency",
)
