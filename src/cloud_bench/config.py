"""Locked config for Bullet 2 (支柱7 TASKS.md) -- the ~$15 vLLM/A100 EAGLE3
speculative-decoding benchmark. Every field here is fixed BEFORE the GPU is
rented, per notes/简历定稿计划-Specter_2026-08-28.md 坑3: cross-framework
acceptance metrics are only comparable when num_speculative_tokens, sampling,
output length and dataset are held identical across arms.

None of this has been run against a real vLLM/GuideLLM install -- neither is
in this repo's .venv, and there is no local CUDA GPU to test against. Treat
the exact CLI flags in orchestrate.py as a best-effort draft from public docs
(2026-08), not a verified recipe; the execution plan
(notes/cloud-bullet2-execution-plan_2026-08-29.md) says to reconcile them
against `vllm serve --help` / `guidellm --help` on the rented box before
spending on the full matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE3_DRAFT_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
# small same-family draft for the "draft_model" arm; verify tokenizer
# compatibility with TARGET_MODEL on the box before relying on this.
DRAFT_MODEL_ARM_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

NUM_SPECULATIVE_TOKENS = 3
TEMPERATURE = 0.0
TOP_P = 1.0
OUTPUT_LEN = 1024
SEED = 42
DATASET = "gsm8k"  # full 1319-example test split -- 坑2, task-domain match matters
CONCURRENCIES = (1, 4, 16, 32, 64)  # 坑1: only a sweep exposes the collapse
ARM_NAMES = ("eagle3", "ngram", "draft_model", "baseline")

VLLM_PORT = 8000
HEALTH_TIMEOUT_S = 300
HEALTH_POLL_INTERVAL_S = 5

# 坑1 (prior_art / plan note): P-EAGLE reports 1.55x@c1 collapsing to 1.05x@c64.
# The sanity check only exercises c=1 against this band -- it is a "did I
# misconfigure something" gate, not a full reproduction.
PAPER_SPEEDUP_LOW = 1.25
PAPER_SPEEDUP_HIGH = 1.32
SANITY_THRESHOLD_FRACTION = 0.95

# Hard cost/time guardrail for orchestrate.run_matrix -- the whole point of
# capping this locally-before-renting is to not leave a GPU running unattended.
DEFAULT_MAX_RUNTIME_MIN = 180


@dataclass(frozen=True)
class ArmSpec:
    name: str
    speculative_config: dict | None  # None => baseline, no --speculative-config


def arm_specs() -> list[ArmSpec]:
    return [
        ArmSpec(
            "eagle3",
            {
                "model": EAGLE3_DRAFT_MODEL,
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                "method": "eagle3",
            },
        ),
        ArmSpec(
            "ngram",
            {
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                "method": "ngram",
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 4,
            },
        ),
        ArmSpec(
            "draft_model",
            {
                "model": DRAFT_MODEL_ARM_MODEL,
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                "method": "draft_model",
            },
        ),
        ArmSpec("baseline", None),
    ]


def arm_spec_by_name(name: str) -> ArmSpec:
    for spec in arm_specs():
        if spec.name == name:
            return spec
    raise KeyError(name)
