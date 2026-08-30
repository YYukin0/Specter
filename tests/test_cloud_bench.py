"""Bullet 2 (支柱7 TASKS.md) local scaffolding -- hermetic tests only.

Neither vllm nor guidellm is installed in this repo's .venv, and there is no
local CUDA GPU. Everything in src/cloud_bench/ talks to those tools only via
subprocess, so all of it is testable by injecting fake popen/run/clock
callables -- these tests never spawn a real subprocess or touch the network.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_bench import config, orchestrate, sanity_check  # noqa: E402


# --------------------------------------------------------------------- config

def test_arm_specs_cover_expected_names():
    names = [s.name for s in config.arm_specs()]
    assert names == list(config.ARM_NAMES)


def test_arm_spec_by_name_roundtrips_and_rejects_unknown():
    assert config.arm_spec_by_name("eagle3").name == "eagle3"
    with pytest.raises(KeyError):
        config.arm_spec_by_name("nope")


# ---------------------------------------------------------------- vllm_serve_cmd

def test_vllm_serve_cmd_eagle3_includes_speculative_config():
    cmd = orchestrate.vllm_serve_cmd(config.arm_spec_by_name("eagle3"))
    assert cmd[:3] == ["vllm", "serve", config.TARGET_MODEL]
    i = cmd.index("--speculative-config")
    spec = json.loads(cmd[i + 1])
    assert spec["model"] == config.EAGLE3_DRAFT_MODEL
    assert spec["method"] == "eagle3"
    assert spec["num_speculative_tokens"] == config.NUM_SPECULATIVE_TOKENS


def test_vllm_serve_cmd_baseline_has_no_speculative_config():
    cmd = orchestrate.vllm_serve_cmd(config.arm_spec_by_name("baseline"))
    assert "--speculative-config" not in cmd


def test_vllm_serve_cmd_locks_seed():
    cmd = orchestrate.vllm_serve_cmd(config.arm_spec_by_name("baseline"))
    assert cmd[cmd.index("--seed") + 1] == str(config.SEED)


def test_vllm_serve_cmd_caps_max_model_len():
    # Without this, vLLM derives max-model-len from the model config (128K
    # for Llama-3.1) and sizes the KV cache/CUDA-graph capture against that,
    # which risks an OOM or a much slower startup for no benefit on GSM8K.
    cmd = orchestrate.vllm_serve_cmd(config.arm_spec_by_name("baseline"))
    assert cmd[cmd.index("--max-model-len") + 1] == str(config.MAX_MODEL_LEN)


# ----------------------------------------------------------------- guidellm_cmd

def test_guidellm_cmd_uses_requested_concurrency_and_locked_config(tmp_path):
    # guidellm==0.7.3 (verified live on the rented A40, 2026-08-30): no
    # `guidellm benchmark` subcommand exists, only `guidellm run` with
    # registry-style `kind=...,key=value` options. See orchestrate.py's
    # guidellm_cmd docstring for the field-by-field trail.
    out = tmp_path / "r.json"
    cmd = orchestrate.guidellm_cmd(16, out)
    assert cmd[0:2] == ["guidellm", "run"]
    profile = cmd[cmd.index("--profile") + 1]
    assert "kind=concurrent" in profile
    assert "streams=16" in profile
    backend = cmd[cmd.index("--backend") + 1]
    assert "kind=openai_http" in backend
    assert f"max_tokens={config.OUTPUT_LEN}" in backend
    assert f"extras.body.temperature={config.TEMPERATURE}" in backend
    assert f"extras.body.top_p={config.TOP_P}" in backend
    assert f"extras.body.seed={config.SEED}" in backend
    data = cmd[cmd.index("--data") + 1]
    assert "kind=huggingface" in data
    assert f"source={config.DATASET}" in data
    assert f"load_kwargs.name={config.DATASET_CONFIG}" in data
    assert f"load_kwargs.split={config.DATASET_SPLIT}" in data
    output = cmd[cmd.index("--output") + 1]
    assert "kind=json" in output
    assert f"path={out}" in output
    constraint = cmd[cmd.index("--constraint") + 1]
    assert "kind=max_duration" in constraint
    assert f"seconds={config.GUIDELLM_MAX_DURATION_S}" in constraint


# --------------------------------------------------------------- wait_for_health

def test_wait_for_health_returns_true_on_first_ok():
    calls = []

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(url, timeout=5):
        calls.append(url)
        return Resp()

    ok = orchestrate.wait_for_health(
        "http://x/health", sleep=lambda s: None, now=lambda: 0.0, urlopen=fake_urlopen
    )
    assert ok is True
    assert calls == ["http://x/health"]


def test_wait_for_health_times_out_when_never_healthy():
    ticks = iter([0, 1, 2, 100])  # last value exceeds the 10s deadline below

    def fake_now():
        return next(ticks)

    def fake_urlopen(url, timeout=5):
        raise OSError("connection refused")

    ok = orchestrate.wait_for_health(
        "http://x/health", timeout_s=10, poll_interval_s=1,
        sleep=lambda s: None, now=fake_now, urlopen=fake_urlopen,
    )
    assert ok is False


# ------------------------------------------------------------- result normalize

def test_normalize_guidellm_result_prefers_nested_metrics_shape():
    raw = {"metrics": {"output_tokens_per_second": {"mean": 42.0},
                        "time_to_first_token_ms": {"p99": 123.0},
                        "time_per_output_token_ms": {"p99": 7.0}}}
    rec = orchestrate.normalize_guidellm_result(raw, concurrency=4)
    assert rec == {
        "concurrency": 4,
        "mean_output_tokens_per_sec": 42.0,
        "ttft_p99_ms": 123.0,
        "tpot_p99_ms": 7.0,
        "mean_acceptance_rate": None,
    }


def test_normalize_guidellm_result_falls_back_to_flat_shape():
    raw = {"output_tokens_per_second": 10.0, "ttft_p99_ms": 5.0}
    rec = orchestrate.normalize_guidellm_result(raw, concurrency=1)
    assert rec["mean_output_tokens_per_sec"] == 10.0
    assert rec["ttft_p99_ms"] == 5.0


# ------------------------------------------------------------------ speedup math

def test_compute_speedup_matches_ratio():
    arm = {1: {"mean_output_tokens_per_sec": 30.0}, 4: {"mean_output_tokens_per_sec": 40.0}}
    base = {1: {"mean_output_tokens_per_sec": 20.0}, 4: {"mean_output_tokens_per_sec": 40.0}}
    out = orchestrate.compute_speedup(arm, base)
    assert out == {1: 1.5, 4: 1.0}


def test_compute_speedup_raises_on_missing_baseline_concurrency():
    with pytest.raises(KeyError):
        orchestrate.compute_speedup({16: {"mean_output_tokens_per_sec": 1.0}}, {})


def test_compute_speedup_raises_on_zero_baseline_throughput():
    with pytest.raises(ValueError):
        orchestrate.compute_speedup(
            {1: {"mean_output_tokens_per_sec": 5.0}},
            {1: {"mean_output_tokens_per_sec": 0.0}},
        )


# --------------------------------------------------------------------- run_matrix

def test_run_matrix_dry_run_builds_full_plan_without_subprocess():
    def boom(*a, **k):
        raise AssertionError("subprocess should not be invoked in dry_run")

    matrix = orchestrate.run_matrix(
        arms=["eagle3", "baseline"], concurrencies=[1, 4],
        dry_run=True, popen=boom, run=boom,
    )
    # 2 arms * (1 serve cmd + 2 concurrencies) = 6 planned commands
    assert len(matrix["plan"]) == 6
    assert matrix["plan"][0][:2] == ["vllm", "serve"]
    assert matrix["results"] == {"eagle3": {"concurrency": {}}, "baseline": {"concurrency": {}}}


def test_run_matrix_executes_and_collects_results(tmp_path, monkeypatch):
    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass

    def fake_popen(cmd):
        return FakeProc()

    def fake_run(cmd, check=True):
        out_path = Path(cmd[cmd.index("--output") + 1].removeprefix("kind=json,path="))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"output_tokens_per_second": 50.0}))

    monkeypatch.setattr(orchestrate, "wait_for_health", lambda *a, **k: True)
    matrix = orchestrate.run_matrix(
        arms=["baseline"], concurrencies=[1],
        results_dir=tmp_path, dry_run=False,
        popen=fake_popen, run=fake_run,
    )
    assert matrix["results"]["baseline"]["concurrency"][1]["mean_output_tokens_per_sec"] == 50.0


def test_run_matrix_aborts_when_health_check_fails(monkeypatch):
    monkeypatch.setattr(orchestrate, "wait_for_health", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="never became healthy"):
        orchestrate.run_matrix(
            arms=["baseline"], concurrencies=[1], dry_run=False,
            popen=lambda cmd: type("P", (), {"terminate": lambda s: None, "wait": lambda s, timeout=None: None})(),
            run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("run should not be called")),
        )


def test_run_matrix_resumes_from_existing_raw_output(tmp_path, monkeypatch):
    # Simulate a prior crashed run: concurrency=1 already has a valid raw
    # output file on disk; concurrency=4 does not.
    cached = tmp_path / "guidellm_baseline_c1.json"
    cached.write_text(json.dumps({"output_tokens_per_second": 50.0}))

    calls = []

    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass

    def fake_run(cmd, check=True):
        out_path = Path(cmd[cmd.index("--output") + 1].removeprefix("kind=json,path="))
        calls.append(out_path.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"output_tokens_per_second": 99.0}))

    monkeypatch.setattr(orchestrate, "wait_for_health", lambda *a, **k: True)
    matrix = orchestrate.run_matrix(
        arms=["baseline"], concurrencies=[1, 4],
        results_dir=tmp_path, dry_run=False,
        popen=lambda cmd: FakeProc(), run=fake_run,
    )
    # concurrency=1 was never re-run -- only concurrency=4 hit `run`
    assert calls == ["guidellm_baseline_c4.json"]
    assert matrix["results"]["baseline"]["concurrency"][1]["mean_output_tokens_per_sec"] == 50.0
    assert matrix["results"]["baseline"]["concurrency"][4]["mean_output_tokens_per_sec"] == 99.0


def test_run_matrix_skips_server_start_when_arm_fully_cached(tmp_path):
    (tmp_path / "guidellm_baseline_c1.json").write_text(json.dumps({"output_tokens_per_second": 50.0}))

    def boom_popen(cmd):
        raise AssertionError("server should not be started when every point is already cached")

    matrix = orchestrate.run_matrix(
        arms=["baseline"], concurrencies=[1],
        results_dir=tmp_path, dry_run=False,
        popen=boom_popen, run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("run should not be called")),
    )
    assert matrix["results"]["baseline"]["concurrency"][1]["mean_output_tokens_per_sec"] == 50.0


def test_run_matrix_rereuns_point_with_corrupt_cached_file(tmp_path, monkeypatch):
    corrupt = tmp_path / "guidellm_baseline_c1.json"
    corrupt.write_text("not valid json {")

    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass

    def fake_run(cmd, check=True):
        out_path = Path(cmd[cmd.index("--output") + 1].removeprefix("kind=json,path="))
        out_path.write_text(json.dumps({"output_tokens_per_second": 77.0}))

    monkeypatch.setattr(orchestrate, "wait_for_health", lambda *a, **k: True)
    matrix = orchestrate.run_matrix(
        arms=["baseline"], concurrencies=[1],
        results_dir=tmp_path, dry_run=False,
        popen=lambda cmd: FakeProc(), run=fake_run,
    )
    assert matrix["results"]["baseline"]["concurrency"][1]["mean_output_tokens_per_sec"] == 77.0


def test_run_matrix_checkpoints_aggregated_results_after_each_point(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.json"
    seen_at_second_call = {}

    class FakeProc:
        def terminate(self): pass
        def wait(self, timeout=None): pass

    def fake_run(cmd, check=True):
        out_path = Path(cmd[cmd.index("--output") + 1].removeprefix("kind=json,path="))
        if out_path.name == "guidellm_baseline_c4.json":
            # by the time the second point runs, the checkpoint from the
            # first point must already be on disk
            seen_at_second_call.update(json.loads(checkpoint.read_text()))
        out_path.write_text(json.dumps({"output_tokens_per_second": 10.0}))

    monkeypatch.setattr(orchestrate, "wait_for_health", lambda *a, **k: True)
    orchestrate.run_matrix(
        arms=["baseline"], concurrencies=[1, 4],
        results_dir=tmp_path, dry_run=False,
        popen=lambda cmd: FakeProc(), run=fake_run,
        checkpoint_path=checkpoint,
    )
    assert checkpoint.exists()
    assert seen_at_second_call["results"]["baseline"]["concurrency"]["1"]["mean_output_tokens_per_sec"] == 10.0


def test_run_matrix_respects_runtime_budget():
    ticks = iter([0, 0, 1000])  # deadline computed from first now(), blown by the third call

    def fake_now():
        return next(ticks)

    with pytest.raises(orchestrate.RuntimeBudgetExceeded):
        orchestrate.run_matrix(
            arms=["baseline"], concurrencies=[1, 4],
            dry_run=True, max_runtime_min=1, now=fake_now,
        )


# --------------------------------------------------------------- demo js / json

def test_to_demo_arms_computes_speedup_relative_to_baseline():
    results = {
        "eagle3": {"concurrency": {1: {"mean_output_tokens_per_sec": 60.0, "ttft_p99_ms": 10.0, "tpot_p99_ms": 2.0}}},
        "baseline": {"concurrency": {1: {"mean_output_tokens_per_sec": 30.0, "ttft_p99_ms": 20.0, "tpot_p99_ms": 4.0}}},
    }
    arms = orchestrate.to_demo_arms(results)
    eagle = next(a for a in arms if a["name"] == "eagle3")
    assert eagle["points"] == [{"concurrency": 1, "speedup": 2.0, "ttft_p99_ms": 10.0, "tpot_p99_ms": 2.0}]


def test_write_results_json_roundtrips(tmp_path):
    path = tmp_path / "sub" / "r.json"
    orchestrate.write_results_json(path, {"plan": [], "results": {}})
    assert json.loads(path.read_text()) == {"plan": [], "results": {}}


def test_write_demo_js_produces_loadable_ready_payload(tmp_path):
    path = tmp_path / "cloud_bench.js"
    results = {
        "eagle3": {"concurrency": {1: {"mean_output_tokens_per_sec": 60.0, "ttft_p99_ms": None, "tpot_p99_ms": None}}},
        "baseline": {"concurrency": {1: {"mean_output_tokens_per_sec": 30.0, "ttft_p99_ms": None, "tpot_p99_ms": None}}},
    }
    orchestrate.write_demo_js(path, results)
    text = path.read_text()
    assert text.startswith("window.SPECTER_CLOUD_BENCH = ")
    payload = json.loads(text[len("window.SPECTER_CLOUD_BENCH = "):].rstrip("\n;").rstrip(";"))
    assert payload["status"] == "ready"
    assert payload["locked_config"]["num_speculative_tokens"] == config.NUM_SPECULATIVE_TOKENS
    names = {a["name"] for a in payload["arms"]}
    assert names == {"eagle3", "baseline"}


# ------------------------------------------------------------------- sanity check

def test_sanity_check_passes_at_paper_low_band():
    ok, msg = sanity_check.check_reproduction(config.PAPER_SPEEDUP_LOW)
    assert ok is True
    assert "PASS" in msg


def test_sanity_check_passes_exactly_at_threshold_boundary():
    threshold = config.PAPER_SPEEDUP_LOW * config.SANITY_THRESHOLD_FRACTION
    ok, _ = sanity_check.check_reproduction(threshold)
    assert ok is True


def test_sanity_check_fails_just_below_threshold():
    threshold = config.PAPER_SPEEDUP_LOW * config.SANITY_THRESHOLD_FRACTION
    ok, msg = sanity_check.check_reproduction(threshold - 0.01)
    assert ok is False
    assert "FAIL" in msg


def test_sanity_check_main_exit_codes(capsys):
    assert sanity_check.main(["--eagle3-tok-per-s", "60", "--baseline-tok-per-s", "40"]) == 0
    assert sanity_check.main(["--eagle3-tok-per-s", "40", "--baseline-tok-per-s", "40"]) == 1
    assert sanity_check.main(["--eagle3-tok-per-s", "40", "--baseline-tok-per-s", "0"]) == 2
