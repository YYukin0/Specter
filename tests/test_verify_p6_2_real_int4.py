"""P6.2 -- real int4 via mlx-lm.

The pure helpers (headline formatting, nan-safe deltas, dir sizing) are checked
without MLX. The end-to-end 4-way comparison is only run when the locally
quantized model dirs exist (they are gitignored build artifacts); otherwise it
skips, so CI without the models stays green.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

vbi = importlib.import_module("verify_p6_2_real_int4")

HAVE_MODELS = vbi.AWQ_DIR.exists() and vbi.GPTQ_DIR.exists()


def test_dir_size_helpers_on_missing_paths():
    assert vbi._dir_gb(Path("/no/such/dir")) == 0.0
    assert vbi._weights_gb("/no/such/dir") == 0.0


def test_headline_marks_degenerate_arm():
    out = {
        "arms": {
            "fp16_mlx": {"decode_tok_s": {"mean": 30.0},
                         "wikitext2_ppl": {"perplexity": 11.5}},
            "mlx_awq_int4_g128": {"decode_tok_s": {"mean": 100.0},
                                  "wikitext2_ppl": {"perplexity": 13.1}},
            "mlx_rtn_int4_g128": {"decode_tok_s": {"mean": 105.0},
                                  "wikitext2_ppl": {"perplexity": 13.8}},
        },
        "disk": {"fp16_mlx": {"weights_gb": 3.09},
                 "mlx_awq_int4_g128": {"weights_gb": 0.84}},
        "deltas_vs_fp16_mlx": {
            "mlx_awq_int4_g128": 1.6,
            "mlx_gptq_int4_g128": None,   # degenerate
            "mlx_rtn_int4_g128": 2.3,
        },
    }
    line = vbi._headline(out)
    assert "GPTQ DEGENERATE" in line
    assert "AWQ +1.600" in line
    assert "3.7x" in line          # 3.09 / 0.84


@pytest.mark.skipif(not HAVE_MODELS, reason="quantized model dirs not present")
def test_smoke_four_way_runs_and_json_is_strict():
    import json

    out = vbi.run(smoke=True)
    assert set(out["arms"]) == {
        "fp16_mlx", "mlx_awq_int4_g128", "mlx_gptq_int4_g128", "mlx_rtn_int4_g128"
    }
    # fp16 + awq + rtn are known-good; gptq is the reproducibly-degenerate arm
    assert out["arms"]["fp16_mlx"]["degenerate"] is False
    assert out["arms"]["mlx_awq_int4_g128"]["degenerate"] is False
    assert out["deltas_vs_fp16_mlx"]["mlx_gptq_int4_g128"] is None
    # strict JSON: no NaN / Infinity tokens
    json.loads(json.dumps(out, allow_nan=False))
