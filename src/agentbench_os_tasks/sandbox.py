"""Throwaway per-task working directories, so setup()/golden_solution()/
verify() can freely read and write files without touching the repo or
colliding with other tasks."""
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .schema import TaskSpec


@contextmanager
def fresh_workdir(task: TaskSpec) -> Generator[Path]:
    """Yields an empty dir with task.setup() already applied."""
    tmp = Path(tempfile.mkdtemp(prefix=f"agentbench_os_{task.task_id}_"))
    try:
        task.setup(tmp)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
