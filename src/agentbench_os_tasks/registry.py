"""
P3.0 task registry -- notes/project_plan_v9.md §7 P3.0 + 附录B, TASKS.md M3.

18 tasks across the 4 附录B categories (file_ops, code_refactor, cli_tools,
multistep_dependency), 4 held out -- one per category, chosen deliberately
as the least-typical member of each category rather than the easiest or an
arbitrary pick (see each held_out_rationale for the specific reasoning; the
short version is "cover every category, and within each category pick the
task most likely to expose a narrow/overfit implementation" so the held-out
set itself isn't systematically biased toward being trivial).

18, i.e. 14 active + 4 held out: the overnight task-queue #2 requires
active (non-held-out) >= 12, and the original design-time set (11 active /
4 held out = 15 total) fell one short. file_ops / code_refactor / cli_tools
each gained one straightforward non-held-out task (_05); the
multistep_dependency category stays at 3 (see tasks/multistep.py's module
docstring for why a 4th there would be padding, not signal). Still well
inside §7 P3.0's 15-20 band and 9.6风险1's 3-5 held-out band. The three
added tasks are active-only, so the held-out set and its selection
rationale are untouched.

**Held-out discipline (notes/project_plan_v9.md §9.6 风险1)**: HELD_OUT_TASKS below is fixed at design
time. Nothing in this repository may run a held-out task's verify() against
an actual model-produced trace, or use a held-out task's pass/fail signal
to adjust GammaTune's hyperparameters (η, δ, γ_min/max, P5.0), until *all*
those hyperparameters are finalized -- and even then, only once (TASKS.md
M8's "held-out 任务集最终跑一次"). This constraint is about *evaluation*
runs, not about testing the verifier code itself: verify_smoke_test.py in
this package runs golden_solution() through every task's verify(),
including held-out ones, because that's checking "is the scoring script
correct," not "how does a model do on this task." See schema.py's TaskSpec
docstring for the same point spelled out at the class level.

P3.0 stops here: this module hands P3.1 a fixed task list and a verify()
per task. It does not run an agent loop, call a model, or estimate
acceptance rate -- that's P3.1 (TASKS.md M3, blocked on P1 as of this
writing per TASKS.md M3's own note).
"""
from __future__ import annotations

from .schema import CATEGORIES, TaskSpec
from .tasks.cli_tools import CLI_TOOLS_TASKS
from .tasks.code_refactor import CODE_REFACTOR_TASKS
from .tasks.file_ops import FILE_OPS_TASKS
from .tasks.multistep import MULTISTEP_TASKS

ALL_TASKS: list[TaskSpec] = [
    *FILE_OPS_TASKS,
    *CODE_REFACTOR_TASKS,
    *CLI_TOOLS_TASKS,
    *MULTISTEP_TASKS,
]

HELD_OUT_TASKS: list[TaskSpec] = [t for t in ALL_TASKS if t.held_out]
ACTIVE_TASKS: list[TaskSpec] = [t for t in ALL_TASKS if not t.held_out]

_TASKS_BY_ID: dict[str, TaskSpec] = {t.task_id: t for t in ALL_TASKS}


def get_task(task_id: str) -> TaskSpec:
    return _TASKS_BY_ID[task_id]


def _validate_registry() -> None:
    ids = [t.task_id for t in ALL_TASKS]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise AssertionError(f"duplicate task_id(s): {dupes}")

    bad_categories = {t.category for t in ALL_TASKS} - set(CATEGORIES)
    if bad_categories:
        raise AssertionError(f"task(s) use unknown categories: {bad_categories}")

    if not (15 <= len(ALL_TASKS) <= 20):
        raise AssertionError(f"expected 15-20 tasks per §7 P3.0, got {len(ALL_TASKS)}")

    if not (3 <= len(HELD_OUT_TASKS) <= 5):
        raise AssertionError(f"expected 3-5 held-out tasks per 9.6风险1, got {len(HELD_OUT_TASKS)}")

    held_out_categories = {t.category for t in HELD_OUT_TASKS}
    missing = set(CATEGORIES) - held_out_categories
    if missing:
        raise AssertionError(f"held-out set doesn't cover every category, missing: {missing}")

    for t in HELD_OUT_TASKS:
        if not t.held_out_rationale:
            raise AssertionError(f"{t.task_id} is held_out but has no held_out_rationale")


_validate_registry()
