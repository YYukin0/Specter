"""P6.5 -- O2 oracle (real model, CPU fp32, greedy-exact).

Part D (specdiff classification contract) is hermetic and always runs. Parts
A/B/C need real Qwen2.5 weights on CPU; they run only when transformers can load
them offline, else skip -- so CI without the models stays green.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

o2 = importlib.import_module("verify_p6_5_o2")


def test_specdiff_backend_nondeterminism_contract():
    """The rule O2 documents: every structural signal equal, only the committed
    prefix differs -> BACKEND_NONDETERMINISM, not a math/cache mechanism."""
    d = o2.part_d_specdiff_contract()
    assert d["contract_holds"]
    assert d["mechanism"] == "BACKEND_NONDETERMINISM"
    assert d["bisect_round"] == 0


def test_greedy_pair_equal_tolerates_budget_overshoot():
    """A trailing length gap <= gamma is not a divergence (the spec decoder
    emits a whole round at once and overshoots max_new_tokens)."""
    class _Res:
        def __init__(self, ids):
            self.token_ids = ids

    import spec_kv
    seen = {}

    def fake_spec(prompt, d, t, tok, **kw):
        return _Res([1, 2, 3, 4, 5, 6])          # 2 past the "budget"

    def fake_target(prompt, t, tok, **kw):
        return _Res([1, 2, 3, 4])

    old = (spec_kv.speculative_generate_kv, spec_kv.target_only_generate_kv)
    o2.speculative_generate_kv = fake_spec
    o2.target_only_generate_kv = fake_target
    try:
        ok, div, s, b = o2._greedy_pair_equal("p", None, None, None,
                                              gamma=4, max_new_tokens=4)
        assert ok and div is None            # gap of 2 <= gamma 4
        # value divergence inside the shared region is still caught
        o2.speculative_generate_kv = lambda *a, **k: _Res([1, 9, 3, 4])
        ok2, div2, _, _ = o2._greedy_pair_equal("p", None, None, None,
                                                gamma=4, max_new_tokens=4)
        assert not ok2 and div2 == 1
    finally:
        o2.speculative_generate_kv, o2.target_only_generate_kv = old


@pytest.mark.skipif(
    not Path.home().joinpath(
        ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct").exists(),
    reason="Qwen2.5 weights not in local HF cache",
)
def test_o2_smoke_end_to_end():
    out = o2.run(smoke=True)
    assert out["A_clean_greedy_exact"]["all_greedy_exact"] is True
    assert out["D_specdiff_classification_contract"]["contract_holds"] is True
    # bitwise batch invariance on the CPU/fp32 reference path
    assert out["C_batch_invariance_logit_delta"]["max_abs_logit_delta"] == 0.0
