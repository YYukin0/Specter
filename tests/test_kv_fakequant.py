"""P7 Track E -- FakeQuantKVCache numerics + the make_draft_cache hook.

Hermetic: hand tensors for the cache, make_fake_pair for the decode-path check.
"""
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_fakequant import FakeQuantKVCache  # noqa: E402
from spec_kv import speculative_generate_kv  # noqa: E402
from spec_oracles import LengthOnlyCache, make_fake_pair  # noqa: E402


def _seq(n, d=8, h=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, h, n, d)


def test_nbits16_is_a_passthrough():
    plain, fq = DynamicCache(), FakeQuantKVCache(nbits=16, residual_len=4)
    for s in range(4):
        k, v = _seq(10, seed=s), _seq(10, seed=s + 100)
        plain.update(k.clone(), v.clone(), 0)
        fq.update(k.clone(), v.clone(), 0)
    assert torch.equal(plain.layers[0].keys, fq.layers[0].keys)
    assert torch.equal(plain.layers[0].values, fq.layers[0].values)


def test_only_the_pre_residual_region_is_quantized():
    fq = FakeQuantKVCache(nbits=4, residual_len=8)
    k, v = _seq(30, seed=1), _seq(30, seed=2)
    fq.update(k.clone(), v.clone(), 0)
    cut = 30 - 8
    # older region: lossy, so it moved
    assert not torch.equal(fq.layers[0].keys[..., :cut, :], k[..., :cut, :])
    assert not torch.equal(fq.layers[0].values[..., :cut, :], v[..., :cut, :])
    # residual window: bit-identical to the fp input
    assert torch.equal(fq.layers[0].keys[..., cut:, :], k[..., cut:, :])
    assert torch.equal(fq.layers[0].values[..., cut:, :], v[..., cut:, :])


def test_crop_stays_inside_the_fp_residual_window():
    fq = FakeQuantKVCache(nbits=4, residual_len=8)
    k, v = _seq(20, seed=3), _seq(20, seed=4)
    fq.update(k.clone(), v.clone(), 0)
    fq.crop(-4)
    assert fq.get_seq_length() == 16
    # positions 12..15 were in the residual window -> untouched fp originals
    assert torch.equal(fq.layers[0].keys[..., 12:16, :], k[..., 12:16, :])


def test_all_zero_input_does_not_nan():
    fq = FakeQuantKVCache(nbits=4, residual_len=2)
    z = torch.zeros(1, 2, 10, 8)
    fq.update(z.clone(), z.clone(), 0)
    assert not torch.isnan(fq.layers[0].keys).any()
    assert not torch.isnan(fq.layers[0].values).any()


def test_more_bits_means_less_quant_error():
    k = _seq(40, seed=5)
    errs = []
    for nb in (4, 8):
        fq = FakeQuantKVCache(nbits=nb, residual_len=4)
        fq.update(k.clone(), k.clone(), 0)
        errs.append((fq.layers[0].keys[..., :36, :] - k[..., :36, :]).abs().max().item())
    assert errs[1] < errs[0]          # 8-bit tighter than 4-bit
    assert errs[0] > 0.0             # 4-bit actually lossy


def test_make_draft_cache_is_output_transparent():
    # passing make_draft_cache explicitly (same type) must not change a token
    draft, target, tok = make_fake_pair()
    a = speculative_generate_kv("hello world", draft, target, tok, gamma=3,
                                max_new_tokens=32, temperature=0.0, seed=1000,
                                make_cache=LengthOnlyCache)
    b = speculative_generate_kv("hello world", draft, target, tok, gamma=3,
                                max_new_tokens=32, temperature=0.0, seed=1000,
                                make_cache=LengthOnlyCache,
                                make_draft_cache=LengthOnlyCache)
    assert a.token_ids == b.token_ids


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
