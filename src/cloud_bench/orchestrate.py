"""Bullet 2 (支柱7 TASKS.md) orchestration -- drives vLLM + GuideLLM through the
locked arm x concurrency matrix on a rented GPU box.

This module never imports vllm or guidellm as Python libraries: everything
goes through subprocess against the `vllm` / `guidellm` CLIs, so it can be
unit-tested on a machine (this one) that has neither installed and no CUDA
GPU. `subprocess.Popen`/`subprocess.run`/the health-check clock are all
injectable parameters for exactly that reason -- see tests/test_cloud_bench.py.

GuideLLM had two CLI syntax generations in the wild; the version actually
installed on the rented box (guidellm==0.7.3, 2026-08-30) only ships the
newer registry-style one -- `guidellm benchmark` does not exist as a
subcommand at all (`guidellm --help` lists only env/export/mock-server/
preprocess/run). `guidellm_cmd` below was reconciled against
`guidellm run --help` plus reading the installed package's pydantic field
definitions directly (backends/openai/http.py, data/deserializers/
huggingface.py, scheduler/strategies.py, benchmark/outputs/serialized.py) on
the actual rented A40 -- see notes/cloud-bullet2-execution-plan_2026-08-29.md
for the verification trail.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from cloud_bench import config


class RuntimeBudgetExceeded(RuntimeError):
    """Raised when run_matrix would exceed its wall-clock budget -- the whole
    point of this guard is to never leave a rented GPU running unattended."""


def vllm_serve_cmd(arm: config.ArmSpec, port: int = config.VLLM_PORT) -> list[str]:
    cmd = [
        "vllm", "serve", config.TARGET_MODEL,
        "--seed", str(config.SEED),
        "--port", str(port),
        "--max-model-len", str(config.MAX_MODEL_LEN),
    ]
    if arm.speculative_config is not None:
        cmd += ["--speculative-config", json.dumps(arm.speculative_config)]
    return cmd


def guidellm_cmd(
    concurrency: int,
    output_path: Path,
    target_url: str | None = None,
    model: str = config.TARGET_MODEL,
) -> list[str]:
    """guidellm==0.7.3 registry-style CLI. Each `--backend`/`--profile`/`--data`/
    `--output`/`--constraint` value is `kind=<type>,key=value,...`; dotted
    keys (e.g. `extras.body.temperature`) build nested dicts (verified via
    `guidellm.utils.arg_string.loads` on the rented box). `max_tokens` sits on
    the backend itself (`OpenAIHTTPBackendArgs.max_tokens`, aliased to
    `max_completion_tokens`) rather than the data source -- the `huggingface`
    data source (unlike `synthetic_text`) has no output-length knob of its
    own, so this is the only place OUTPUT_LEN gets enforced for a real
    dataset. temperature/top_p/seed ride along in `extras.body`, which is
    merged into the outgoing OpenAI-style request body.

    Without an explicit `--constraint`, `guidellm run` has no default cap and
    will walk the full 1319-row GSM8K test split -- discovered live on the
    rented A40 when a concurrency=1 baseline run was still going 8+ minutes
    in (~34 tok/s generation, 1024 max output tokens => hours to exhaust the
    split). `MaxDurationConstraintArgs.seconds` (guidellm.scheduler.
    constraints, cross-checked against `guidellm run --help`'s `--constraint
    kind=max_duration,...`) bounds each arm x concurrency point to a fixed
    wall-clock budget instead -- 坑25, matches the 60s/point figure the
    execution plan called out before any of this was verified."""
    target_url = target_url or f"http://localhost:{config.VLLM_PORT}"
    backend = (
        f"kind=openai_http,target={target_url},model={model}"
        f",max_tokens={config.OUTPUT_LEN}"
        f",extras.body.temperature={config.TEMPERATURE}"
        f",extras.body.top_p={config.TOP_P}"
        f",extras.body.seed={config.SEED}"
    )
    profile = f"kind=concurrent,streams={concurrency}"
    data = (
        f"kind=huggingface,source={config.DATASET}"
        f",load_kwargs.name={config.DATASET_CONFIG}"
        f",load_kwargs.split={config.DATASET_SPLIT}"
    )
    output = f"kind=json,path={output_path}"
    constraint = f"kind=max_duration,seconds={config.GUIDELLM_MAX_DURATION_S}"
    return [
        "guidellm", "run",
        "--backend", backend,
        "--profile", profile,
        "--data", data,
        "--output", output,
        "--constraint", constraint,
    ]


def wait_for_health(
    url: str,
    timeout_s: int = config.HEALTH_TIMEOUT_S,
    poll_interval_s: int = config.HEALTH_POLL_INTERVAL_S,
    sleep=time.sleep,
    now=time.monotonic,
    urlopen=urllib.request.urlopen,
) -> bool:
    deadline = now() + timeout_s
    while now() < deadline:
        try:
            with urlopen(url, timeout=5) as resp:
                if getattr(resp, "status", 200) == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        sleep(poll_interval_s)
    return False


def normalize_guidellm_result(raw: dict, concurrency: int) -> dict:
    """Extract the fields we need from a guidellm==0.7.3 `run` JSON output.

    Schema verified live against a real `guidellm run` output on the rented
    A40 (2026-08-30, results/cloud_bench_raw/guidellm_baseline_c1.json) --
    the earlier version of this function was a placeholder guessing at
    field paths that never matched, silently producing all-null records
    (caught because compute_speedup would have divided by None, not because
    anything crashed -- 坑25 continued). Real shape:
    `raw = {"metadata", "config", "benchmarks": [...]}`; each
    `benchmarks[i]["metrics"][<metric_name>]` is a per-completion-status
    breakdown (`{"successful": <stats>, "errored": <stats>, ...}`), and each
    `<stats>` is a full distribution object (`mean`, `median`, `min`, `max`,
    `count`, ... and a nested `percentiles` dict for `p99` etc. -- p99 is
    NOT a top-level key on the stats object itself, an earlier version of
    this function got that wrong and silently returned None for every
    percentile field). GuideLLM measures serving performance only -- there
    is no acceptance-rate metric in its schema at all, so that field is
    always None here (vLLM's own `/metrics` Prometheus endpoint would be the
    place to get it, not guidellm)."""
    benchmarks = raw.get("benchmarks") or [{}]
    metrics = benchmarks[0].get("metrics", {}) if benchmarks else {}

    def stat(metric_name: str, stat_name: str = "mean"):
        bucket = metrics.get(metric_name, {})
        successful = bucket.get("successful") if isinstance(bucket, dict) else None
        if not isinstance(successful, dict):
            return None
        if stat_name.startswith("p") and stat_name[1:].isdigit():
            return successful.get("percentiles", {}).get(stat_name)
        return successful.get(stat_name)

    return {
        "concurrency": concurrency,
        "mean_output_tokens_per_sec": stat("output_tokens_per_second"),
        "ttft_p99_ms": stat("time_to_first_token_ms", "p99"),
        "tpot_p99_ms": stat("time_per_output_token_ms", "p99"),
        "mean_acceptance_rate": None,
    }


def compute_speedup(arm_records: dict[int, dict], baseline_records: dict[int, dict]) -> dict[int, float]:
    out = {}
    for c, rec in arm_records.items():
        if c not in baseline_records:
            raise KeyError(f"baseline missing concurrency={c}")
        b = baseline_records[c].get("mean_output_tokens_per_sec")
        a = rec.get("mean_output_tokens_per_sec")
        if not b:
            raise ValueError(f"baseline throughput is zero/missing at concurrency={c}")
        out[c] = a / b
    return out


def run_matrix(
    arms: list[str] | None = None,
    concurrencies: list[int] | None = None,
    results_dir: Path = Path("results/cloud_bench_raw"),
    dry_run: bool = True,
    max_runtime_min: float = config.DEFAULT_MAX_RUNTIME_MIN,
    popen=subprocess.Popen,
    run=subprocess.run,
    sleep=time.sleep,
    now=time.monotonic,
    checkpoint_path: Path | None = None,
) -> dict:
    """Serves each arm in turn, drives GuideLLM across the concurrency sweep,
    tears the server down, and moves to the next arm. dry_run=True (default)
    builds every command and returns it in "plan" without executing anything
    -- that's the mode this repo's tests exercise, and the mode to run once
    on the box before spending on the real matrix.

    Resumable by construction: each concurrency point's raw GuideLLM output
    lands at results_dir/guidellm_{arm}_c{c}.json as soon as that point
    finishes, and a point whose file already exists (and parses) is reused
    instead of re-run. If the whole arm is already done, its vllm server is
    never even started. If checkpoint_path is given, the aggregated matrix
    is rewritten to that path after every point completes, so a process that
    dies mid-run (dropped SSH, preempted instance) loses at most the point it
    was on -- rerunning the same command picks up where it left off."""
    arms = arms or [s.name for s in config.arm_specs()]
    concurrencies = concurrencies or list(config.CONCURRENCIES)
    deadline = now() + max_runtime_min * 60

    plan: list[list[str]] = []
    results: dict[str, dict] = {}
    for arm_name in arms:
        spec = config.arm_spec_by_name(arm_name)
        serve_cmd = vllm_serve_cmd(spec)
        plan.append(serve_cmd)
        arm_result: dict = {"concurrency": {}}
        out_paths = {c: results_dir / f"guidellm_{arm_name}_c{c}.json" for c in concurrencies}
        arm_already_done = (not dry_run) and all(_load_cached(p) is not None for p in out_paths.values())
        proc = None
        try:
            if not dry_run and not arm_already_done:
                proc = popen(serve_cmd)
                if not wait_for_health(f"http://localhost:{config.VLLM_PORT}/health", sleep=sleep, now=now):
                    raise RuntimeError(f"vllm server for arm={arm_name} never became healthy")
            for c in concurrencies:
                if now() > deadline:
                    raise RuntimeBudgetExceeded(
                        f"max_runtime_min={max_runtime_min} exceeded before arm={arm_name} concurrency={c} -- "
                        "aborting so the rented instance doesn't run unattended"
                    )
                out_path = out_paths[c]
                cmd = guidellm_cmd(c, out_path)
                plan.append(cmd)
                if not dry_run:
                    raw = _load_cached(out_path)
                    if raw is None:
                        run(cmd, check=True)
                        raw = json.loads(out_path.read_text())
                    arm_result["concurrency"][c] = normalize_guidellm_result(raw, c)
                    if checkpoint_path is not None:
                        write_results_json(checkpoint_path, {"plan": plan, "results": {**results, arm_name: arm_result}})
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        results[arm_name] = arm_result
    return {"plan": plan, "results": results}


def _load_cached(out_path: Path) -> dict | None:
    """Returns the parsed contents of a prior GuideLLM output file, or None
    if it doesn't exist or is unreadable (partial write from a killed run) --
    either way the caller re-runs that point rather than trusting a corrupt
    file."""
    if not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def to_demo_arms(matrix_results: dict[str, dict], baseline_name: str = "baseline") -> list[dict]:
    baseline = matrix_results[baseline_name]["concurrency"]
    arms_out = []
    for name, data in matrix_results.items():
        points = []
        for c, rec in sorted(data["concurrency"].items()):
            b = baseline.get(c)
            if not b or not b.get("mean_output_tokens_per_sec"):
                continue
            speedup = rec["mean_output_tokens_per_sec"] / b["mean_output_tokens_per_sec"]
            points.append({
                "concurrency": c,
                "speedup": speedup,
                "ttft_p99_ms": rec.get("ttft_p99_ms"),
                "tpot_p99_ms": rec.get("tpot_p99_ms"),
            })
        arms_out.append({"name": name, "points": points})
    return arms_out


def write_results_json(path: Path, matrix: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, indent=2, sort_keys=True))


def write_demo_js(path: Path, matrix_results: dict) -> None:
    payload = {
        "status": "ready",
        "target_model": config.TARGET_MODEL,
        "draft_model": config.EAGLE3_DRAFT_MODEL,
        "locked_config": {
            "num_speculative_tokens": config.NUM_SPECULATIVE_TOKENS,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "output_len": config.OUTPUT_LEN,
            "dataset": config.DATASET,
            "concurrencies": list(config.CONCURRENCIES),
        },
        "arms": to_demo_arms(matrix_results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.SPECTER_CLOUD_BENCH = " + json.dumps(payload, indent=2) + ";\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--execute", dest="dry_run", action="store_false")
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--concurrencies", nargs="*", type=int, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results/cloud_bench_raw"))
    p.add_argument("--results-json", type=Path, default=Path("results/bullet2_vllm_eagle3.json"))
    p.add_argument("--demo-js", type=Path, default=None,
                    help="also write docs/site/cloud_bench.js in the 'ready' shape")
    p.add_argument("--max-runtime-min", type=float, default=config.DEFAULT_MAX_RUNTIME_MIN)
    args = p.parse_args(argv)

    matrix = run_matrix(
        arms=args.arms, concurrencies=args.concurrencies,
        results_dir=args.results_dir, dry_run=args.dry_run,
        max_runtime_min=args.max_runtime_min,
        checkpoint_path=None if args.dry_run else args.results_json,
    )
    if args.dry_run:
        print(f"dry run -- {len(matrix['plan'])} commands planned, nothing executed:")
        for cmd in matrix["plan"]:
            print(" ", " ".join(cmd))
        return 0

    write_results_json(args.results_json, matrix)
    print(f"wrote {args.results_json}")
    if args.demo_js:
        write_demo_js(args.demo_js, matrix["results"])
        print(f"wrote {args.demo_js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
