"""
P7 Track E -- hand-rolled fake-quantization of the target KV cache.

No optimum-quanto / hqq / optimum: none are installed and there is no py3.14
wheel for them (same constraint as note 05's fake-quant vs real int4). This is a
*simulation* of the lossy signal an int-N KV cache carries, not a packed
representation -- nothing here saves a byte of memory. What it reproduces is the
numerical error: per-channel symmetric int-N quant -> dequant, stored back as
fp16, applied to every KV position older than a fp "residual" window.

Subclasses `transformers.DynamicCache`, so it drops straight into
`speculative_step_kv` / `target_only_generate_kv` as their `make_cache`.

`residual_len` (default 64) is the count of most-recent positions kept in full
fp16. Any speculative rollback here is `crop(-g)` with `g <= gamma <= 8`, far
inside that window, so `crop()` never lands on quantized data and there is no
"requantize a partially-rolled-back block" hazard (Pitfall 37 / plan 坑E-2).
"""
import torch
from transformers import DynamicCache


class FakeQuantKVCache(DynamicCache):
    def __init__(self, nbits: int = 4, residual_len: int = 64, group_size: int = 64,
                 quantize_keys: bool = True, quantize_values: bool = True):
        super().__init__()
        self.nbits = nbits
        self.residual_len = residual_len
        self.group_size = group_size          # reserved; per-channel below is group=head_dim
        self.quantize_keys = quantize_keys
        self.quantize_values = quantize_values

    def _fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_heads, seq, head_dim]. Per-channel (last-dim) symmetric int-N:
        one scale per (batch, head, channel), computed over the sequence axis."""
        if self.nbits >= 16:
            return x
        qmax = (1 << (self.nbits - 1)) - 1
        amax = x.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = amax / qmax
        q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
        return (q * scale).to(x.dtype)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        k, v = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if self.nbits < 16:
            seq = k.shape[-2]
            if seq > self.residual_len:
                cut = seq - self.residual_len
                layer = self.layers[layer_idx]
                if self.quantize_keys:
                    layer.keys[..., :cut, :] = self._fake_quant(layer.keys[..., :cut, :])
                if self.quantize_values:
                    layer.values[..., :cut, :] = self._fake_quant(layer.values[..., :cut, :])
        return k, v

    # crop() is inherited unchanged: the residual fp window (>> gamma) guarantees
    # a speculative rollback only ever trims full-precision tail positions, so it
    # never has to requantize a partially-cropped block (Pitfall 37 / 坑E-2).
