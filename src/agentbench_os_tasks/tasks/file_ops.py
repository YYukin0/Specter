"""
文件操作类 (附录B category 1). 5 tasks, 1 held-out (file_ops_04).

Config format note: the plan's own example ("config.yaml ... debug: true ->
false") uses YAML, but this repo has no YAML dependency (requirements.txt is
empty) and P3.0 shouldn't introduce one just for a toy fixture -- JSON says
the same thing about "flip a boolean field, preserve the rest" without a new
dependency. See PR notes for this call.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import TaskSpec, VerifyResult

# ---------------------------------------------------------------------------
# file_ops_01 -- flip a boolean config field, leave everything else alone
# ---------------------------------------------------------------------------

_CONFIG_ORIGINAL = {"debug": True, "retries": 3, "name": "svc"}


def _fo01_setup(workdir: Path) -> None:
    (workdir / "config.json").write_text(json.dumps(_CONFIG_ORIGINAL, indent=2) + "\n")


def _fo01_golden(workdir: Path) -> None:
    path = workdir / "config.json"
    data = json.loads(path.read_text())
    data["debug"] = False
    path.write_text(json.dumps(data, indent=2) + "\n")


def _fo01_verify(workdir: Path) -> VerifyResult:
    path = workdir / "config.json"
    if not path.exists():
        return VerifyResult(False, "config.json missing")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"config.json is not valid JSON: {e}")
    if data.get("debug") is not False:
        return VerifyResult(False, f"expected debug=false, got {data.get('debug')!r}")
    for key in ("retries", "name"):
        if data.get(key) != _CONFIG_ORIGINAL[key]:
            return VerifyResult(False, f"{key} changed: expected {_CONFIG_ORIGINAL[key]!r}, got {data.get(key)!r}")
    return VerifyResult(True, "debug=false, other fields unchanged")


FILE_OPS_01 = TaskSpec(
    task_id="file_ops_01",
    category="file_ops",
    description=(
        "Read `config.json` in the task directory. Set the `debug` field to "
        "`false` while leaving every other field unchanged. Save the file "
        "in place as valid JSON."
    ),
    setup=_fo01_setup,
    golden_solution=_fo01_golden,
    verify=_fo01_verify,
)


# ---------------------------------------------------------------------------
# file_ops_02 -- rename .txt -> .md recursively, preserve content, leave
# other extensions alone
# ---------------------------------------------------------------------------

_FO02_TXT_FILES = {
    "notes/a.txt": "alpha content\n",
    "notes/sub/b.txt": "beta content\n",
}
_FO02_OTHER_FILES = {
    "notes/c.log": "not a txt file, leave me alone\n",
}


def _fo02_setup(workdir: Path) -> None:
    for rel, content in {**_FO02_TXT_FILES, **_FO02_OTHER_FILES}.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _fo02_golden(workdir: Path) -> None:
    for rel in _FO02_TXT_FILES:
        src = workdir / rel
        dst = src.with_suffix(".md")
        src.rename(dst)


def _fo02_verify(workdir: Path) -> VerifyResult:
    for rel, content in _FO02_TXT_FILES.items():
        old = workdir / rel
        new = old.with_suffix(".md")
        if old.exists():
            return VerifyResult(False, f"{rel} should have been renamed away")
        if not new.exists():
            return VerifyResult(False, f"{new.relative_to(workdir)} missing")
        if new.read_text() != content:
            return VerifyResult(False, f"{new.relative_to(workdir)} content changed")
    for rel, content in _FO02_OTHER_FILES.items():
        p = workdir / rel
        if not p.exists() or p.read_text() != content:
            return VerifyResult(False, f"{rel} should have been left untouched")
    return VerifyResult(True, "all .txt renamed to .md, other files untouched")


FILE_OPS_02 = TaskSpec(
    task_id="file_ops_02",
    category="file_ops",
    description=(
        "Rename every `.txt` file under `notes/` (including subdirectories) "
        "to the same name with a `.md` extension, keeping directory "
        "structure and content unchanged. Do not touch files with other "
        "extensions."
    ),
    setup=_fo02_setup,
    golden_solution=_fo02_golden,
    verify=_fo02_verify,
)


# ---------------------------------------------------------------------------
# file_ops_03 -- normalize CRLF/CR line endings to LF, content unchanged
# ---------------------------------------------------------------------------

_FO03_FILES = {
    "logs/x.txt": b"line one\r\nline two\r\nline three\n",
    "logs/y.txt": b"only\rcarriage\rreturns\r",
}


def _fo03_setup(workdir: Path) -> None:
    for rel, content in _FO03_FILES.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def _normalize(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _fo03_golden(workdir: Path) -> None:
    for rel in _FO03_FILES:
        p = workdir / rel
        p.write_bytes(_normalize(p.read_bytes()))


def _fo03_verify(workdir: Path) -> VerifyResult:
    for rel, original in _FO03_FILES.items():
        p = workdir / rel
        if not p.exists():
            return VerifyResult(False, f"{rel} missing")
        data = p.read_bytes()
        if b"\r" in data:
            return VerifyResult(False, f"{rel} still contains \\r")
        if data != _normalize(original):
            return VerifyResult(False, f"{rel} text content changed beyond line endings")
    return VerifyResult(True, "all line endings normalized to LF, text unchanged")


FILE_OPS_03 = TaskSpec(
    task_id="file_ops_03",
    category="file_ops",
    description=(
        "Normalize line endings to LF (`\\n`) in every `.txt` file under "
        "`logs/`. Do not change the text content otherwise."
    ),
    setup=_fo03_setup,
    golden_solution=_fo03_golden,
    verify=_fo03_verify,
)


# ---------------------------------------------------------------------------
# file_ops_04 (HELD-OUT) -- add a field to every object in a JSON array,
# re-save with a specific formatting convention (indent + sorted keys)
# ---------------------------------------------------------------------------

_FO04_RECORDS = [
    {"id": 1, "name": "alice"},
    {"id": 2, "name": "bob", "role": "admin"},
    {"id": 3, "name": "carol"},
]


def _fo04_setup(workdir: Path) -> None:
    (workdir / "records.json").write_text(json.dumps(_FO04_RECORDS) + "\n")


def _fo04_golden(workdir: Path) -> None:
    path = workdir / "records.json"
    data = json.loads(path.read_text())
    for record in data:
        record["schema_version"] = 2
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _fo04_verify(workdir: Path) -> VerifyResult:
    path = workdir / "records.json"
    if not path.exists():
        return VerifyResult(False, "records.json missing")
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return VerifyResult(False, f"records.json is not valid JSON: {e}")
    if len(data) != len(_FO04_RECORDS):
        return VerifyResult(False, f"expected {len(_FO04_RECORDS)} records, got {len(data)}")
    for original, record in zip(_FO04_RECORDS, data):
        if record.get("schema_version") != 2:
            return VerifyResult(False, f"record {original['id']} missing schema_version=2")
        for key, value in original.items():
            if record.get(key) != value:
                return VerifyResult(False, f"record {original['id']} field {key!r} changed")
    expected_text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if text != expected_text:
        return VerifyResult(False, "file is not formatted as indent=2, sort_keys=True JSON")
    return VerifyResult(True, "schema_version added to all records, formatting matches spec")


FILE_OPS_04 = TaskSpec(
    task_id="file_ops_04",
    category="file_ops",
    description=(
        "Read `records.json` (a JSON array of objects). Add a field "
        '`"schema_version": 2` to every object, preserving existing fields. '
        "Save back to `records.json` as pretty-printed JSON (2-space "
        "indent) with each object's keys sorted alphabetically."
    ),
    setup=_fo04_setup,
    golden_solution=_fo04_golden,
    verify=_fo04_verify,
    held_out=True,
    held_out_rationale=(
        "One of two file_ops variants that require exact output "
        "formatting (not just correct values) -- the more corner-case "
        "flavored of the two, picked over file_ops_01/02/03 specifically "
        "so the held-out set isn't systematically the 'easy' member of "
        "its category (see registry.py module docstring for the full "
        "held-out selection rationale)."
    ),
)


# ---------------------------------------------------------------------------
# file_ops_05 -- delete backup files (name ends in .bak) under build/, leave
# every other file alone, including one whose name merely contains ".bak"
# ---------------------------------------------------------------------------

_FO05_BAK_FILES = {
    "build/app.js.bak": "stale build artifact\n",
    "build/lib/util.js.bak": "stale util artifact\n",
}
_FO05_KEEP_FILES = {
    "build/app.js": "current build\n",
    "build/lib/util.js": "current util\n",
    "build/notes.bak.txt": "the .bak here is not the extension -- keep me\n",
}


def _fo05_setup(workdir: Path) -> None:
    for rel, content in {**_FO05_BAK_FILES, **_FO05_KEEP_FILES}.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _fo05_golden(workdir: Path) -> None:
    for rel in _FO05_BAK_FILES:
        (workdir / rel).unlink()


def _fo05_verify(workdir: Path) -> VerifyResult:
    for rel in _FO05_BAK_FILES:
        if (workdir / rel).exists():
            return VerifyResult(False, f"{rel} should have been deleted")
    for rel, content in _FO05_KEEP_FILES.items():
        p = workdir / rel
        if not p.exists():
            return VerifyResult(False, f"{rel} should have been left in place")
        if p.read_text() != content:
            return VerifyResult(False, f"{rel} content changed")
    return VerifyResult(True, "all *.bak under build/ deleted, everything else intact")


FILE_OPS_05 = TaskSpec(
    task_id="file_ops_05",
    category="file_ops",
    description=(
        "Delete every file whose name ends in `.bak` under `build/` "
        "(including subdirectories). Leave all other files untouched, "
        "including any file whose name merely contains `.bak` earlier on "
        "(e.g. `notes.bak.txt`)."
    ),
    setup=_fo05_setup,
    golden_solution=_fo05_golden,
    verify=_fo05_verify,
)


FILE_OPS_TASKS: list[TaskSpec] = [
    FILE_OPS_01,
    FILE_OPS_02,
    FILE_OPS_03,
    FILE_OPS_04,
    FILE_OPS_05,
]
