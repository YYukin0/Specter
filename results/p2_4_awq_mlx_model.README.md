# P2.4 mlx_lm.awq quantized model artifact — NOT in git

The P2.4 cross-check (`results/p2_4_mlx_awq_crosscheck.json`) was produced from a
4-bit AWQ model quantized by `mlx_lm.awq` from `mlx-community/Qwen2.5-3B-Instruct-bf16`.

**The quantized weights directory is a build artifact and is deliberately not
committed:**

| | |
|---|---|
| local path | `src/results/p2_4_awq_mlx_model/` |
| size | **1.6 GB** (`du -sh`, 2026-08-28) |
| how to regenerate | `python src/p2_4_mlx_awq_crosscheck.py --phase quantize` (~530 s wall on this Mac) |
| config | bits=4, group_size=128, num_samples=32, seq_len=256, n_grid=20, seed=123 |

Only the JSON result (`p2_4_mlx_awq_crosscheck.json`) is tracked. `src/results/`
is gitignored (see `.gitignore`); the authoritative copies of the small B-track
result JSONs now live at the repo top-level `results/`.
