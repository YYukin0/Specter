"""
多步骤依赖类 (附录B category 4). 3 tasks, 1 held-out (multistep_03) -- fewer
than the other three categories' 4 apiece. Honest accounting of why, per the
"don't pad to hit a round number" instruction:

This category turned out to be the hardest of the four to design well
*given that P3.0 has no engine yet* (P3.1 is what will actually run a model
against these). The intended property is "step 2's output must be derived
from step 1's output, not recomputed from scratch" -- but a verify()
function here only ever sees the final filesystem state, the same as every
other task in this package. It cannot inspect whether the agent's tool
calls actually read the intermediate file, only whether the final numbers
are *consistent with* having done so. That's a materially weaker guarantee
than what "test long-context reliance on a prior tool call's output" is
supposed to measure.

The mitigation used here: every fixture is constructed so that computing
the final artifact from *all* raw input (skipping the intermediate
filtering/subsetting step) gives a numerically different, and therefore
verifiably wrong, answer from computing it from just the correct
intermediate subset. verify() recomputes both the intermediate artifact and
the final artifact independently from the original fixture (never from the
agent's own intermediate file) and checks both against the agent's output.
This catches "skipped step 1, hallucinated step 2" and "step 1 wrong,
step 2 didn't propagate the error" -- but it is still a filesystem-state
proxy, not a trace-level check. A real trace-level check (e.g. asserting
the agent's tool-call log actually read the intermediate file before
writing the final one) needs P3.1's engine to exist first; tracked as a
possible P3.1 enhancement, not solved here.

Adding a 4th multistep task to match the other categories' count would have
meant either accepting a weaker fixture (final answer reachable without the
intermediate step) or a contrived 3-hop chain that stops looking like a
plausible bash/tool-call task and starts looking like a puzzle -- both
worse than shipping 3 tasks that actually hold the discriminating property
above.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..schema import TaskSpec, VerifyResult

# ---------------------------------------------------------------------------
# multistep_01 -- filter logs containing a marker into candidates.txt, then
# sum line counts of *only those* candidates into summary.json
# ---------------------------------------------------------------------------

_MS01_FILES = {
    "logs/svc1.log": "FATAL crash\nline2\nline3\nline4\n",   # FATAL, 4 lines
    "logs/svc2.log": "info one\ninfo two\n",                  # no FATAL, 2 lines
    "logs/svc3.log": "warn\nFATAL now\nend\n",                # FATAL, 3 lines
    "logs/svc4.log": "a\nb\nc\nd\ne\n",                       # no FATAL, 5 lines
    "logs/svc5.log": "FATAL\n",                               # FATAL, 1 line
}


def _ms01_setup(workdir: Path) -> None:
    for rel, content in _MS01_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _ms01_expected_candidates(workdir: Path) -> list[str]:
    root = workdir / "logs"
    return sorted(p.name for p in root.iterdir() if "FATAL" in p.read_text())


def _ms01_expected_total_lines(workdir: Path, candidates: list[str]) -> int:
    root = workdir / "logs"
    return sum(len((root / name).read_text().splitlines()) for name in candidates)


def _ms01_golden(workdir: Path) -> None:
    candidates = _ms01_expected_candidates(workdir)
    (workdir / "candidates.txt").write_text("\n".join(candidates) + "\n")
    total = _ms01_expected_total_lines(workdir, candidates)
    (workdir / "summary.json").write_text(json.dumps({"total_lines": total}))


def _ms01_verify(workdir: Path) -> VerifyResult:
    cand_path = workdir / "candidates.txt"
    summary_path = workdir / "summary.json"
    if not cand_path.exists():
        return VerifyResult(False, "candidates.txt missing")
    if not summary_path.exists():
        return VerifyResult(False, "summary.json missing")

    got_candidates = [line for line in cand_path.read_text().splitlines() if line]
    expected_candidates = _ms01_expected_candidates(workdir)
    if got_candidates != expected_candidates:
        return VerifyResult(False, f"candidates.txt = {got_candidates}, expected {expected_candidates}")

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"summary.json not valid JSON: {e}")
    expected_total = _ms01_expected_total_lines(workdir, expected_candidates)
    if summary.get("total_lines") != expected_total:
        return VerifyResult(
            False,
            f"total_lines={summary.get('total_lines')!r}, expected {expected_total} "
            "(derived strictly from candidates.txt's FATAL-flagged files)",
        )
    return VerifyResult(True, "candidates correct, summary derived from candidates only")


MULTISTEP_01 = TaskSpec(
    task_id="multistep_01",
    category="multistep_dependency",
    description=(
        "Step 1: under `logs/`, find every file that contains the literal "
        "string `FATAL` anywhere in its content. Write their filenames "
        "(not full paths), sorted alphabetically, one per line, to "
        "`candidates.txt`. Step 2: for each filename listed in "
        "`candidates.txt`, count its total number of lines, and write the "
        'sum of all those line counts to `summary.json` as `{"total_lines": '
        "<N>}`. `summary.json` must be computed strictly from the files "
        "listed in `candidates.txt`, not all files in `logs/`."
    ),
    setup=_ms01_setup,
    golden_solution=_ms01_golden,
    verify=_ms01_verify,
)


# ---------------------------------------------------------------------------
# multistep_02 -- filter CSV rows by a boolean column into filtered.csv,
# then aggregate stats from *that* into stats.json
# ---------------------------------------------------------------------------

_MS02_CSV = """id,name,score,active
1,alice,80,true
2,bob,50,false
3,carol,90,true
4,dave,10,false
5,eve,70,true
"""


def _ms02_setup(workdir: Path) -> None:
    (workdir / "records.csv").write_text(_MS02_CSV)


def _ms02_expected_filtered_rows(workdir: Path) -> list[dict]:
    reader = csv.DictReader(io.StringIO((workdir / "records.csv").read_text()))
    return [row for row in reader if row["active"] == "true"]


def _ms02_golden(workdir: Path) -> None:
    rows = _ms02_expected_filtered_rows(workdir)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "name", "score", "active"])
    writer.writeheader()
    writer.writerows(rows)
    (workdir / "filtered.csv").write_text(buf.getvalue())

    count = len(rows)
    avg = round(sum(float(r["score"]) for r in rows) / count, 2) if count else 0.0
    (workdir / "stats.json").write_text(json.dumps({"count": count, "avg_score": avg}))


def _ms02_verify(workdir: Path) -> VerifyResult:
    filtered_path = workdir / "filtered.csv"
    stats_path = workdir / "stats.json"
    if not filtered_path.exists():
        return VerifyResult(False, "filtered.csv missing")
    if not stats_path.exists():
        return VerifyResult(False, "stats.json missing")

    got_rows = list(csv.DictReader(io.StringIO(filtered_path.read_text())))
    expected_rows = _ms02_expected_filtered_rows(workdir)
    if got_rows != expected_rows:
        return VerifyResult(False, f"filtered.csv rows = {got_rows}, expected {expected_rows}")

    try:
        stats = json.loads(stats_path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"stats.json not valid JSON: {e}")
    expected_count = len(expected_rows)
    expected_avg = round(sum(float(r["score"]) for r in expected_rows) / expected_count, 2)
    if stats.get("count") != expected_count:
        return VerifyResult(False, f"count={stats.get('count')!r}, expected {expected_count}")
    got_avg = stats.get("avg_score")
    if not isinstance(got_avg, (int, float)) or abs(got_avg - expected_avg) > 1e-6:
        return VerifyResult(
            False,
            f"avg_score={got_avg!r}, expected {expected_avg} "
            "(derived strictly from filtered.csv, not records.csv)",
        )
    return VerifyResult(True, "filtered.csv correct, stats derived from filtered.csv only")


MULTISTEP_02 = TaskSpec(
    task_id="multistep_02",
    category="multistep_dependency",
    description=(
        "Step 1: read `records.csv`. Keep only rows where the `active` "
        "column is `true` (case-sensitive), and write them (with the same "
        "header) to `filtered.csv`. Step 2: from `filtered.csv`, compute "
        "the count of rows and the average of the `score` column (rounded "
        'to 2 decimals), and write to `stats.json` as `{"count": <N>, '
        '"avg_score": <X>}`. `stats.json` must be derived from '
        "`filtered.csv`, not from the original `records.csv`."
    ),
    setup=_ms02_setup,
    golden_solution=_ms02_golden,
    verify=_ms02_verify,
)


# ---------------------------------------------------------------------------
# multistep_03 (HELD-OUT) -- flag files importing a deprecated module, then
# classify each flagged file by whether it carries a migration marker
# ---------------------------------------------------------------------------

_MS03_FILES = {
    "project/a.py": "import legacy_mod\n# TODO: migrate\nprint('a')\n",   # flagged, compliant
    "project/b.py": "import legacy_mod\nprint('b')\n",                    # flagged, non-compliant
    "project/c.py": "import os\nprint('c')\n",                            # not flagged
    "project/d.py": "import legacy_mod\n# TODO: migrate\nprint('d')\n",   # flagged, compliant
}


def _ms03_setup(workdir: Path) -> None:
    for rel, content in _MS03_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _ms03_expected_flagged(workdir: Path) -> list[str]:
    root = workdir / "project"
    return sorted(
        p.name for p in root.iterdir()
        if any(line.strip() == "import legacy_mod" for line in p.read_text().splitlines())
    )


def _ms03_expected_report(workdir: Path, flagged: list[str]) -> dict:
    root = workdir / "project"
    compliant, non_compliant = [], []
    for name in flagged:
        lines = (root / name).read_text().splitlines()
        if any(line.strip() == "# TODO: migrate" for line in lines):
            compliant.append(name)
        else:
            non_compliant.append(name)
    return {"compliant": sorted(compliant), "non_compliant": sorted(non_compliant)}


def _ms03_golden(workdir: Path) -> None:
    flagged = _ms03_expected_flagged(workdir)
    (workdir / "flagged.txt").write_text("\n".join(flagged) + "\n")
    report = _ms03_expected_report(workdir, flagged)
    (workdir / "report.json").write_text(json.dumps(report, sort_keys=True))


def _ms03_verify(workdir: Path) -> VerifyResult:
    flagged_path = workdir / "flagged.txt"
    report_path = workdir / "report.json"
    if not flagged_path.exists():
        return VerifyResult(False, "flagged.txt missing")
    if not report_path.exists():
        return VerifyResult(False, "report.json missing")

    got_flagged = [line for line in flagged_path.read_text().splitlines() if line]
    expected_flagged = _ms03_expected_flagged(workdir)
    if got_flagged != expected_flagged:
        return VerifyResult(False, f"flagged.txt = {got_flagged}, expected {expected_flagged}")

    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"report.json not valid JSON: {e}")
    expected_report = _ms03_expected_report(workdir, expected_flagged)
    got_report = {
        "compliant": sorted(report.get("compliant", [])),
        "non_compliant": sorted(report.get("non_compliant", [])),
    }
    if got_report != expected_report:
        return VerifyResult(
            False,
            f"report.json = {got_report}, expected {expected_report} "
            "(derived strictly from flagged.txt's files)",
        )
    return VerifyResult(True, "flagged.txt correct, report derived from flagged.txt only")


MULTISTEP_03 = TaskSpec(
    task_id="multistep_03",
    category="multistep_dependency",
    description=(
        "Step 1: under `project/`, find every `.py` file that contains "
        "the exact line `import legacy_mod` (after stripping whitespace). "
        "Write their filenames, sorted alphabetically, one per line, to "
        "`flagged.txt`. Step 2: for each filename in `flagged.txt`, check "
        "whether that file also contains the exact line "
        "`# TODO: migrate`. Write a report to `report.json` as "
        '`{"compliant": [...], "non_compliant": [...]}`, where '
        "`compliant` lists filenames (sorted) that have the marker and "
        "`non_compliant` lists filenames (sorted) that don't. Only "
        "consider files from `flagged.txt`, not all files under "
        "`project/`."
    ),
    setup=_ms03_setup,
    golden_solution=_ms03_golden,
    verify=_ms03_verify,
    held_out=True,
    held_out_rationale=(
        "The longest dependency chain in this category (find -> "
        "per-item classify -> aggregate into two buckets) and the one "
        "most likely to be sensitive to exactly the kind of long-context "
        "gamma tuning P5.0/P5.1 will iterate on -- holding it out "
        "protects the task most representative of what 风险1 is actually "
        "worried about, not just an arbitrary pick."
    ),
)


MULTISTEP_TASKS: list[TaskSpec] = [MULTISTEP_01, MULTISTEP_02, MULTISTEP_03]
