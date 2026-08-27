"""
代码重构类 (附录B category 2). 5 tasks, 1 held-out (code_refactor_04).

Every task bundles its own plain-assert test script (no pytest/unittest --
this repo has no test-framework dependency and P3.0 shouldn't add one for a
toy fixture). verify() runs the bundled test script with a subprocess and
checks the exit code; a failing assert raises, which is a nonzero exit.
verify() also hashes the test script's own content to catch an agent
"passing" the task by editing the test instead of the code.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..schema import TaskSpec, VerifyResult


def _run_test_script(workdir: Path, script_name: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, script_name],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]


def _test_script_unchanged(workdir: Path, script_name: str, original: str) -> bool:
    path = workdir / script_name
    return path.exists() and path.read_text() == original


# ---------------------------------------------------------------------------
# code_refactor_01 -- §13 walkthrough task: parse_config supports nested
# dot-separated keys, existing (partly-failing) test file must all pass
# ---------------------------------------------------------------------------

_CR01_UTILS = '''def parse_config(config, key):
    return config[key]
'''

_CR01_TEST = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import parse_config


def test_flat_key():
    cfg = {"a": 1, "b": 2}
    assert parse_config(cfg, "a") == 1


def test_nested_key():
    cfg = {"a": {"b": {"c": 42}}}
    assert parse_config(cfg, "a.b.c") == 42


def test_nested_key_missing_raises():
    cfg = {"a": {"b": 1}}
    try:
        parse_config(cfg, "a.b.c")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


if __name__ == "__main__":
    test_flat_key()
    test_nested_key()
    test_nested_key_missing_raises()
    print("OK")
'''

_CR01_GOLDEN_UTILS = '''def parse_config(config, key):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node
'''


def _cr01_setup(workdir: Path) -> None:
    (workdir / "utils.py").write_text(_CR01_UTILS)
    (workdir / "test_utils.py").write_text(_CR01_TEST)


def _cr01_golden(workdir: Path) -> None:
    (workdir / "utils.py").write_text(_CR01_GOLDEN_UTILS)


def _cr01_verify(workdir: Path) -> VerifyResult:
    if not _test_script_unchanged(workdir, "test_utils.py", _CR01_TEST):
        return VerifyResult(False, "test_utils.py was modified -- not allowed")
    ok, output = _run_test_script(workdir, "test_utils.py")
    if not ok:
        return VerifyResult(False, f"test_utils.py failed:\n{output}")
    return VerifyResult(True, "test_utils.py passed, test file untouched")


CODE_REFACTOR_01 = TaskSpec(
    task_id="code_refactor_01",
    category="code_refactor",
    description=(
        "In `utils.py`, refactor `parse_config(config, key)` so `key` can "
        'be a dot-separated path into nested dicts (e.g. `"a.b.c"`), while '
        "still supporting plain top-level keys. Missing keys (nested or "
        "not) should raise `KeyError`. Run `python test_utils.py` and make "
        "it pass; do not modify `test_utils.py`."
    ),
    setup=_cr01_setup,
    golden_solution=_cr01_golden,
    verify=_cr01_verify,
    design_notes=(
        "This is the concrete instantiation of the §13 end-to-end "
        "walkthrough task (project_plan_v9.md §13 第7点: 'FINISHED 决定了这条 "
        "trace 在下游任务指标里怎么计分')."
    ),
)


# ---------------------------------------------------------------------------
# code_refactor_02 -- rename a function across its definition and call
# sites, behavior (and CLI output) unchanged
# ---------------------------------------------------------------------------

_CR02_MATHUTILS = '''def add_two(a, b):
    return a + b
'''

_CR02_MAIN = '''import json
from mathutils import add_two

if __name__ == "__main__":
    r1 = add_two(2, 3)
    r2 = add_two(10, 20)
    print(json.dumps({"r1": r1, "r2": r2}))
'''

_CR02_GOLDEN_MATHUTILS = '''def add(a, b):
    return a + b
'''

_CR02_GOLDEN_MAIN = '''import json
from mathutils import add

if __name__ == "__main__":
    r1 = add(2, 3)
    r2 = add(10, 20)
    print(json.dumps({"r1": r1, "r2": r2}))
'''

_CR02_EXPECTED_STDOUT = {"r1": 5, "r2": 30}


def _cr02_setup(workdir: Path) -> None:
    (workdir / "mathutils.py").write_text(_CR02_MATHUTILS)
    (workdir / "main.py").write_text(_CR02_MAIN)


def _cr02_golden(workdir: Path) -> None:
    (workdir / "mathutils.py").write_text(_CR02_GOLDEN_MATHUTILS)
    (workdir / "main.py").write_text(_CR02_GOLDEN_MAIN)


def _cr02_verify(workdir: Path) -> VerifyResult:
    for py_file in workdir.glob("*.py"):
        if "add_two" in py_file.read_text():
            return VerifyResult(False, f"{py_file.name} still references add_two")
    mathutils_src = (workdir / "mathutils.py").read_text()
    if "def add(" not in mathutils_src:
        return VerifyResult(False, "mathutils.py does not define add(...)")
    proc = subprocess.run(
        [sys.executable, "main.py"], cwd=workdir, capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return VerifyResult(False, f"main.py crashed:\n{proc.stderr[-2000:]}")
    import json as _json

    try:
        stdout_json = _json.loads(proc.stdout.strip())
    except _json.JSONDecodeError:
        return VerifyResult(False, f"main.py stdout not valid JSON: {proc.stdout!r}")
    if stdout_json != _CR02_EXPECTED_STDOUT:
        return VerifyResult(False, f"main.py output {stdout_json} != expected {_CR02_EXPECTED_STDOUT}")
    return VerifyResult(True, "add_two renamed to add everywhere, behavior unchanged")


CODE_REFACTOR_02 = TaskSpec(
    task_id="code_refactor_02",
    category="code_refactor",
    description=(
        "Rename the function `add_two` to `add` everywhere it is defined "
        "or called in this directory (`mathutils.py`, `main.py`). Do not "
        "change its behavior. After the rename, running `python main.py` "
        "must still print the same JSON output it does today."
    ),
    setup=_cr02_setup,
    golden_solution=_cr02_golden,
    verify=_cr02_verify,
)


# ---------------------------------------------------------------------------
# code_refactor_03 -- extract duplicated logic into a named helper,
# structural check via ast (not just behavior parity)
# ---------------------------------------------------------------------------

_CR03_STATS = '''def summarize_a(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def summarize_b(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
'''

_CR03_TEST = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from stats import summarize_a, summarize_b


def test_summarize_a():
    assert summarize_a([1, 2, 3]) == {"mean": 2.0, "min": 1, "max": 3}


def test_summarize_b():
    assert summarize_b([4, 5, 9]) == {"mean": 6.0, "min": 4, "max": 9}


if __name__ == "__main__":
    test_summarize_a()
    test_summarize_b()
    print("OK")
'''

_CR03_GOLDEN_STATS = '''def _summarize(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def summarize_a(values):
    return _summarize(values)


def summarize_b(values):
    return _summarize(values)
'''


def _cr03_setup(workdir: Path) -> None:
    (workdir / "stats.py").write_text(_CR03_STATS)
    (workdir / "test_stats.py").write_text(_CR03_TEST)


def _cr03_golden(workdir: Path) -> None:
    (workdir / "stats.py").write_text(_CR03_GOLDEN_STATS)


def _cr03_verify(workdir: Path) -> VerifyResult:
    if not _test_script_unchanged(workdir, "test_stats.py", _CR03_TEST):
        return VerifyResult(False, "test_stats.py was modified -- not allowed")
    ok, output = _run_test_script(workdir, "test_stats.py")
    if not ok:
        return VerifyResult(False, f"test_stats.py failed:\n{output}")

    import ast

    tree = ast.parse((workdir / "stats.py").read_text())
    defined_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "_summarize" not in defined_names:
        return VerifyResult(False, "no function named _summarize is defined")

    def calls_summarize(func_name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "_summarize":
                        return True
        return False

    if not calls_summarize("summarize_a") or not calls_summarize("summarize_b"):
        return VerifyResult(False, "summarize_a/summarize_b do not both call _summarize")
    return VerifyResult(True, "duplicated logic extracted into _summarize, both callers use it, tests pass")


CODE_REFACTOR_03 = TaskSpec(
    task_id="code_refactor_03",
    category="code_refactor",
    description=(
        "`stats.py` has two functions, `summarize_a` and `summarize_b`, "
        "that each compute {mean, min, max} of a list using duplicated "
        "logic. Extract the shared logic into a new helper function named "
        "exactly `_summarize` and have both `summarize_a` and "
        "`summarize_b` call it. Keep their public behavior (arguments, "
        "return values) unchanged. Run `python test_stats.py`; it must "
        "still pass."
    ),
    setup=_cr03_setup,
    golden_solution=_cr03_golden,
    verify=_cr03_verify,
    design_notes=(
        "Behavior-only tests can't tell 'deduped' apart from 'left "
        "duplicated' -- verify() also does an ast-based structural check "
        "that _summarize exists and is called from both sites."
    ),
)


# ---------------------------------------------------------------------------
# code_refactor_04 (HELD-OUT) -- add type hints only, behavior unchanged,
# structural check via ast on the annotated signature
# ---------------------------------------------------------------------------

_CR04_FORMATTER = '''def format_price(amount, currency="USD"):
    return f"{currency} {amount:.2f}"
'''

_CR04_TEST = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from formatter import format_price


def test_default_currency():
    assert format_price(9.5) == "USD 9.50"


def test_explicit_currency():
    assert format_price(3, "EUR") == "EUR 3.00"


if __name__ == "__main__":
    test_default_currency()
    test_explicit_currency()
    print("OK")
'''

_CR04_GOLDEN_FORMATTER = '''def format_price(amount: float, currency: str = "USD") -> str:
    return f"{currency} {amount:.2f}"
'''


def _cr04_setup(workdir: Path) -> None:
    (workdir / "formatter.py").write_text(_CR04_FORMATTER)
    (workdir / "test_formatter.py").write_text(_CR04_TEST)


def _cr04_golden(workdir: Path) -> None:
    (workdir / "formatter.py").write_text(_CR04_GOLDEN_FORMATTER)


def _cr04_verify(workdir: Path) -> VerifyResult:
    if not _test_script_unchanged(workdir, "test_formatter.py", _CR04_TEST):
        return VerifyResult(False, "test_formatter.py was modified -- not allowed")
    ok, output = _run_test_script(workdir, "test_formatter.py")
    if not ok:
        return VerifyResult(False, f"test_formatter.py failed:\n{output}")

    import ast

    tree = ast.parse((workdir / "formatter.py").read_text())
    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "format_price"),
        None,
    )
    if func is None:
        return VerifyResult(False, "format_price is no longer defined")
    if func.returns is None or getattr(func.returns, "id", None) != "str":
        return VerifyResult(False, "return type annotation is not `str`")
    args = func.args.args
    if len(args) != 2:
        return VerifyResult(False, f"expected 2 parameters, got {len(args)}")
    amount_arg, currency_arg = args
    if amount_arg.annotation is None or getattr(amount_arg.annotation, "id", None) != "float":
        return VerifyResult(False, "`amount` is not annotated as float")
    if currency_arg.annotation is None or getattr(currency_arg.annotation, "id", None) != "str":
        return VerifyResult(False, "`currency` is not annotated as str")
    if not func.args.defaults or ast.literal_eval(func.args.defaults[-1]) != "USD":
        return VerifyResult(False, "`currency` default value changed from \"USD\"")
    return VerifyResult(True, "type hints added, default preserved, behavior unchanged")


CODE_REFACTOR_04 = TaskSpec(
    task_id="code_refactor_04",
    category="code_refactor",
    description=(
        "Add type hints to `format_price`'s signature in `formatter.py`: "
        "`amount` is `float`, `currency` is `str` with default `\"USD\"`, "
        "and the return type is `str`. Do not change the function's "
        "behavior. Run `python test_formatter.py`; it must still pass."
    ),
    setup=_cr04_setup,
    golden_solution=_cr04_golden,
    verify=_cr04_verify,
    held_out=True,
    held_out_rationale=(
        "The other three code_refactor tasks all require an actual "
        "logic change; this one is annotation-only, which is a "
        "meaningfully different failure mode (an agent could satisfy the "
        "behavior test while leaving hints untouched) -- worth holding "
        "out specifically because it's the odd one out, not because it's "
        "the easiest."
    ),
)


# ---------------------------------------------------------------------------
# code_refactor_05 -- replace a repeated magic number with a named module
# constant, behavior unchanged, ast check that the constant exists and is
# referenced from both call sites (behavior test alone can't see "still a
# literal" vs "now a shared constant")
# ---------------------------------------------------------------------------

_CR05_PRICING = '''def with_tax(amount):
    return round(amount * 1.08, 2)


def tax_only(amount):
    return round(amount * 1.08 - amount, 2)
'''

_CR05_TEST = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pricing import with_tax, tax_only


def test_with_tax():
    assert with_tax(100) == 108.0


def test_tax_only():
    assert tax_only(100) == 8.0


if __name__ == "__main__":
    test_with_tax()
    test_tax_only()
    print("OK")
'''

_CR05_GOLDEN_PRICING = '''TAX_RATE = 1.08


def with_tax(amount):
    return round(amount * TAX_RATE, 2)


def tax_only(amount):
    return round(amount * TAX_RATE - amount, 2)
'''


def _cr05_setup(workdir: Path) -> None:
    (workdir / "pricing.py").write_text(_CR05_PRICING)
    (workdir / "test_pricing.py").write_text(_CR05_TEST)


def _cr05_golden(workdir: Path) -> None:
    (workdir / "pricing.py").write_text(_CR05_GOLDEN_PRICING)


def _cr05_verify(workdir: Path) -> VerifyResult:
    if not _test_script_unchanged(workdir, "test_pricing.py", _CR05_TEST):
        return VerifyResult(False, "test_pricing.py was modified -- not allowed")
    ok, output = _run_test_script(workdir, "test_pricing.py")
    if not ok:
        return VerifyResult(False, f"test_pricing.py failed:\n{output}")

    import ast

    src = (workdir / "pricing.py").read_text()
    tree = ast.parse(src)
    const_names = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    if "TAX_RATE" not in const_names:
        return VerifyResult(False, "no module-level constant named TAX_RATE")

    # the 1.08 literal may survive only in the module-level TAX_RATE assignment,
    # never inside a function body
    for func in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for n in ast.walk(func):
            if isinstance(n, ast.Constant) and n.value == 1.08:
                return VerifyResult(False, f"literal 1.08 still used inside {func.name}()")

    def references_tax_rate(func_name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return any(
                    isinstance(n, ast.Name) and n.id == "TAX_RATE" for n in ast.walk(node)
                )
        return False

    if not references_tax_rate("with_tax") or not references_tax_rate("tax_only"):
        return VerifyResult(False, "with_tax/tax_only do not both reference TAX_RATE")
    return VerifyResult(True, "magic number replaced by TAX_RATE, used at both sites, tests pass")


CODE_REFACTOR_05 = TaskSpec(
    task_id="code_refactor_05",
    category="code_refactor",
    description=(
        "`pricing.py` hard-codes the tax rate `1.08` as a literal in both "
        "`with_tax` and `tax_only`. Introduce a module-level constant "
        "named exactly `TAX_RATE` (value `1.08`) and use it in both "
        "functions instead of the literal, so the number `1.08` appears "
        "only in the constant's definition. Keep behavior identical. Run "
        "`python test_pricing.py`; it must still pass."
    ),
    setup=_cr05_setup,
    golden_solution=_cr05_golden,
    verify=_cr05_verify,
)


CODE_REFACTOR_TASKS: list[TaskSpec] = [
    CODE_REFACTOR_01,
    CODE_REFACTOR_02,
    CODE_REFACTOR_03,
    CODE_REFACTOR_04,
    CODE_REFACTOR_05,
]
