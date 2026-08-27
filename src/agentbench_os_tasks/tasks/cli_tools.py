"""
命令行工具调用类 (附录B category 3). 5 tasks, 1 held-out (cli_04).

These describe the kind of query a real agent would answer with
grep/find/awk-ish shell tool calls; verify() recomputes the ground truth
independently from the same fixture files rather than trusting whatever
logic golden_solution() used, so a bug shared between golden_solution and
verify can't silently cancel out.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..schema import TaskSpec, VerifyResult

# ---------------------------------------------------------------------------
# cli_01 -- count lines matching a substring across a directory, JSON out
# ---------------------------------------------------------------------------

_CLI01_FILES = {
    "logs/a.log": "INFO start\nERROR boom\nINFO done\n",
    "logs/b.log": "ERROR again\nERROR twice\n",
    "logs/c.log": "INFO fine\n",
}


def _cli01_setup(workdir: Path) -> None:
    for rel, content in _CLI01_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _cli01_expected_count(workdir: Path) -> int:
    return sum(
        line.count("ERROR") > 0
        for rel in _CLI01_FILES
        for line in (workdir / rel).read_text().splitlines()
        if "ERROR" in line
    )


def _cli01_golden(workdir: Path) -> None:
    count = _cli01_expected_count(workdir)
    (workdir / "result.json").write_text(json.dumps({"pattern": "ERROR", "count": count}))


def _cli01_verify(workdir: Path) -> VerifyResult:
    path = workdir / "result.json"
    if not path.exists():
        return VerifyResult(False, "result.json missing")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"result.json not valid JSON: {e}")
    expected = _cli01_expected_count(workdir)
    if data.get("pattern") != "ERROR":
        return VerifyResult(False, f"pattern field wrong: {data.get('pattern')!r}")
    if data.get("count") != expected:
        return VerifyResult(False, f"count {data.get('count')!r} != expected {expected}")
    return VerifyResult(True, f"count matches expected {expected}")


CLI_01 = TaskSpec(
    task_id="cli_01",
    category="cli_tools",
    description=(
        "Count how many lines across all files under `logs/` contain the "
        "substring `ERROR` (case-sensitive). Write the result to "
        '`result.json` as `{"pattern": "ERROR", "count": <N>}`.'
    ),
    setup=_cli01_setup,
    golden_solution=_cli01_golden,
    verify=_cli01_verify,
)


# ---------------------------------------------------------------------------
# cli_02 -- find files by extension, sorted relative-path listing
# ---------------------------------------------------------------------------

_CLI02_FILES = {
    "project/main.py": "print('hi')\n",
    "project/pkg/util.py": "def f(): pass\n",
    "project/pkg/README.md": "docs\n",
    "project/data.json": "{}\n",
}


def _cli02_setup(workdir: Path) -> None:
    for rel, content in _CLI02_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _cli02_expected_lines(workdir: Path) -> list[str]:
    root = workdir / "project"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.py"))


def _cli02_golden(workdir: Path) -> None:
    lines = _cli02_expected_lines(workdir)
    (workdir / "py_files.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def _cli02_verify(workdir: Path) -> VerifyResult:
    path = workdir / "py_files.txt"
    if not path.exists():
        return VerifyResult(False, "py_files.txt missing")
    got = [line for line in path.read_text().splitlines() if line]
    expected = _cli02_expected_lines(workdir)
    if got != expected:
        return VerifyResult(False, f"got {got}, expected {expected}")
    return VerifyResult(True, "py_files.txt matches expected sorted listing")


CLI_02 = TaskSpec(
    task_id="cli_02",
    category="cli_tools",
    description=(
        "List every `.py` file under `project/` (including "
        "subdirectories), and write their paths relative to `project/`, "
        "one per line, sorted alphabetically, to `py_files.txt`."
    ),
    setup=_cli02_setup,
    golden_solution=_cli02_golden,
    verify=_cli02_verify,
)


# ---------------------------------------------------------------------------
# cli_03 -- sum a CSV column, JSON out
# ---------------------------------------------------------------------------

_CLI03_CSV = """id,category,amount
1,food,12.50
2,rent,900.00
3,food,7.25
4,transport,15.00
"""


def _cli03_setup(workdir: Path) -> None:
    (workdir / "data.csv").write_text(_CLI03_CSV)


def _cli03_expected_sum(workdir: Path) -> float:
    reader = csv.DictReader(io.StringIO((workdir / "data.csv").read_text()))
    return round(sum(float(row["amount"]) for row in reader), 2)


def _cli03_golden(workdir: Path) -> None:
    total = _cli03_expected_sum(workdir)
    (workdir / "total.json").write_text(json.dumps({"column": "amount", "sum": total}))


def _cli03_verify(workdir: Path) -> VerifyResult:
    path = workdir / "total.json"
    if not path.exists():
        return VerifyResult(False, "total.json missing")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"total.json not valid JSON: {e}")
    if data.get("column") != "amount":
        return VerifyResult(False, f"column field wrong: {data.get('column')!r}")
    expected = _cli03_expected_sum(workdir)
    got = data.get("sum")
    if not isinstance(got, (int, float)) or abs(got - expected) > 1e-6:
        return VerifyResult(False, f"sum {got!r} != expected {expected}")
    return VerifyResult(True, f"sum matches expected {expected}")


CLI_03 = TaskSpec(
    task_id="cli_03",
    category="cli_tools",
    description=(
        "Read `data.csv` (has a header row). Sum the `amount` column and "
        'write the result to `total.json` as `{"column": "amount", "sum": '
        "<total>}`."
    ),
    setup=_cli03_setup,
    golden_solution=_cli03_golden,
    verify=_cli03_verify,
)


# ---------------------------------------------------------------------------
# cli_04 (HELD-OUT) -- per-file substring line counts as a JSON dict,
# including zero-count files
# ---------------------------------------------------------------------------

_CLI04_FILES = {
    "logs/svc1.log": "WARN disk low\nINFO ok\nWARN disk low again\n",
    "logs/svc2.log": "INFO all good\n",
    "logs/svc3.log": "WARN retry\n",
}


def _cli04_setup(workdir: Path) -> None:
    for rel, content in _CLI04_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _cli04_expected_counts(workdir: Path) -> dict[str, int]:
    root = workdir / "logs"
    return {
        p.name: sum(1 for line in p.read_text().splitlines() if "WARN" in line)
        for p in sorted(root.iterdir())
    }


def _cli04_golden(workdir: Path) -> None:
    counts = _cli04_expected_counts(workdir)
    (workdir / "warn_counts.json").write_text(json.dumps(counts, sort_keys=True))


def _cli04_verify(workdir: Path) -> VerifyResult:
    path = workdir / "warn_counts.json"
    if not path.exists():
        return VerifyResult(False, "warn_counts.json missing")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"warn_counts.json not valid JSON: {e}")
    expected = _cli04_expected_counts(workdir)
    if data != expected:
        return VerifyResult(False, f"got {data}, expected {expected}")
    return VerifyResult(True, "per-file WARN counts match, zero-count file included")


CLI_04 = TaskSpec(
    task_id="cli_04",
    category="cli_tools",
    description=(
        "For every file under `logs/`, count how many lines contain the "
        "substring `WARN`. Write a JSON object to `warn_counts.json` "
        "mapping each file's name (not full path) to its count, including "
        "files with a count of 0."
    ),
    setup=_cli04_setup,
    golden_solution=_cli04_golden,
    verify=_cli04_verify,
    held_out=True,
    held_out_rationale=(
        "Deliberately includes a zero-count file, the kind of boundary "
        "case that a proportionally-shrunk or hastily-written task set "
        "tends to drop (see memory note on smoke-test boundary "
        "conditions) -- worth protecting from tuning pressure precisely "
        "because it's the edge case, not the modal one."
    ),
)


# ---------------------------------------------------------------------------
# cli_05 -- identify the file with the most lines under a dir, JSON out
# (exactly one file wins; verify recomputes the winner independently)
# ---------------------------------------------------------------------------

_CLI05_FILES = {
    "docs/intro.md": "a\nb\nc\n",
    "docs/guide.md": "one\ntwo\nthree\nfour\nfive\n",
    "docs/faq.md": "just one line\n",
}


def _cli05_setup(workdir: Path) -> None:
    for rel, content in _CLI05_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _cli05_expected(workdir: Path) -> dict:
    root = workdir / "docs"
    counts = {p.name: len(p.read_text().splitlines()) for p in sorted(root.iterdir())}
    top = max(counts, key=lambda name: counts[name])
    return {"file": top, "lines": counts[top]}


def _cli05_golden(workdir: Path) -> None:
    (workdir / "biggest.json").write_text(json.dumps(_cli05_expected(workdir)))


def _cli05_verify(workdir: Path) -> VerifyResult:
    path = workdir / "biggest.json"
    if not path.exists():
        return VerifyResult(False, "biggest.json missing")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"biggest.json not valid JSON: {e}")
    expected = _cli05_expected(workdir)
    if data != expected:
        return VerifyResult(False, f"got {data}, expected {expected}")
    return VerifyResult(True, f"correctly identified {expected['file']} ({expected['lines']} lines)")


CLI_05 = TaskSpec(
    task_id="cli_05",
    category="cli_tools",
    description=(
        "Under `docs/`, find the file with the greatest number of lines. "
        'Write `{"file": "<name>", "lines": <N>}` to `biggest.json`, where '
        "`<name>` is the file's name (not its full path). Exactly one file "
        "has the most lines."
    ),
    setup=_cli05_setup,
    golden_solution=_cli05_golden,
    verify=_cli05_verify,
)


CLI_TOOLS_TASKS: list[TaskSpec] = [CLI_01, CLI_02, CLI_03, CLI_04, CLI_05]
