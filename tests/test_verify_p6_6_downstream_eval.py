"""支柱7 Bullet 3 -- downstream eval (GSM8K + IFEval) orchestrator.

Pure helpers only: metric selection, delta computation, server-log request
counting. The end-to-end run needs the isolated .venv-lmeval harness, the mlx-lm
server and the quantised model dirs, so it is never exercised here.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

dse = importlib.import_module("verify_p6_6_downstream_eval")


def test_pick_metrics_gsm8k_keeps_value_and_stderr():
    res = {
        "exact_match,flexible-extract": 0.42,
        "exact_match_stderr,flexible-extract": 0.03,
        "exact_match,strict-match": 0.11,
        "exact_match_stderr,strict-match": 0.02,
        "alias": "gsm8k",
    }
    out = dse.pick_metrics("gsm8k", res)
    assert out["exact_match,flexible-extract"] == 0.42
    assert out["exact_match_stderr,flexible-extract"] == 0.03
    assert out["exact_match,strict-match"] == 0.11
    assert "alias" not in out


def test_pick_metrics_ignores_absent_keys():
    out = dse.pick_metrics("ifeval", {"prompt_level_strict_acc,none": 0.3})
    assert out == {"prompt_level_strict_acc,none": 0.3}


def test_compute_deltas_quantised_minus_fp16():
    arms = {
        "fp16": {"tasks": {"gsm8k": {"metrics": {
            "exact_match,flexible-extract": 0.50,
            "exact_match_stderr,flexible-extract": 0.03}}}},
        "self_awq": {"tasks": {"gsm8k": {"metrics": {
            "exact_match,flexible-extract": 0.44,
            "exact_match_stderr,flexible-extract": 0.03}}}},
    }
    d = dse.compute_deltas(arms, ["gsm8k"], ["fp16", "self_awq"])
    assert d["self_awq"]["gsm8k"] == {"exact_match,flexible-extract": -0.06}
    assert "fp16" not in d


def test_compute_deltas_skips_errored_task():
    arms = {
        "fp16": {"tasks": {"ifeval": {"metrics": {"prompt_level_strict_acc,none": 0.3}}}},
        "self_awq": {"tasks": {"ifeval": {"error": "boom"}}},
    }
    d = dse.compute_deltas(arms, ["ifeval"], ["fp16", "self_awq"])
    assert d["self_awq"] == {}


def test_server_request_count(tmp_path):
    log = tmp_path / "s.log"
    log.write_text(
        'x\n127.0.0.1 - - "POST /v1/chat/completions HTTP/1.1" 200 -\n'
        'noise\n127.0.0.1 - - "POST /v1/chat/completions HTTP/1.1" 200 -\n')
    assert dse.server_request_count(log) == 2
    assert dse.server_request_count(tmp_path / "missing.log") == -1


def test_arms_registry_shape():
    a = dse.arms()
    assert set(a) == {"fp16", "self_awq", "mlx_awq_int4"}
    assert all(isinstance(v, Path) for v in a.values())
