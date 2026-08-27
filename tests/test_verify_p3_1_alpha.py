"""FakeModel smoke for src/verify_p3_1_alpha.py.

No real weights: a position-dependent one-hot FakeModel (same shape as
tests/test_gammatune.py) drives speculative_generate through the P3.1
helpers. Checks the group runner / summarizer / verdict wiring, not any
acceptance-rate science (that needs the real model pair).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rejection_sampling  # noqa: E402
from verify_p3_1_alpha import _run_group, _summarize_group, _verdict  # noqa: E402


class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self, argmax_at, vocab=8):
        self.argmax_at = argmax_at
        self.vocab = vocab
        self.generation_config = type("g", (), {"eos_token_id": None})()

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids):
        seqlen = input_ids.shape[1]
        logits = torch.full((1, seqlen, self.vocab), -10.0)
        for p in range(seqlen):
            logits[0, p, self.argmax_at(p)] = 10.0
        return _Out(logits)


class _Tok:
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def _ctx(tokenizer, prompt, device, apply_chat_template):
    return torch.zeros((1, 2), dtype=torch.long)


def test_p3_1_helpers_run_and_verdict_is_well_formed(monkeypatch):
    monkeypatch.setattr(rejection_sampling, "encode_prompt", _ctx)

    # draft == target everywhere -> every draft accepted -> alpha == 1.0 in both groups
    draft = FakeModel(lambda p: 1)
    target = FakeModel(lambda p: 1)

    struct_rows = _run_group("structured", ["do X", "do Y"], draft, target, _Tok(), [0, 1],
                             gamma=3, temperature=0.0, max_new_tokens=12)
    free_rows = _run_group("freetext", ["prose a", "prose b"], draft, target, _Tok(), [0, 1],
                           gamma=3, temperature=0.0, max_new_tokens=12)

    assert len(struct_rows) == 4 and len(free_rows) == 4
    s_sum, f_sum = _summarize_group(struct_rows), _summarize_group(free_rows)
    assert s_sum["pooled_alpha"] == 1.0 and f_sum["pooled_alpha"] == 1.0
    assert s_sum["evaluated_total"] > 0

    v = _verdict(s_sum, f_sum)
    assert v["verdict"] == "no_clear_difference"          # identical groups
    assert abs(v["pooled_alpha_gap_structured_minus_freetext"]) < 1e-9
    assert set(v) >= {"verdict", "per_run_mean_gap", "combined_std", "threshold"}


def test_verdict_flags_higher_when_gap_is_real():
    struct = {"pooled_alpha": 0.90, "per_run_alpha": {"mean": 0.90, "std": 0.01}}
    free = {"pooled_alpha": 0.70, "per_run_alpha": {"mean": 0.70, "std": 0.01}}
    assert _verdict(struct, free)["verdict"] == "structured_alpha_higher"

    # same means, huge overlap -> not significant
    noisy = {"pooled_alpha": 0.72, "per_run_alpha": {"mean": 0.72, "std": 0.30}}
    assert _verdict(struct, noisy)["verdict"] == "no_clear_difference"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
