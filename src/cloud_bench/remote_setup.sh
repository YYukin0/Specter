#!/usr/bin/env bash
# Bullet 2 (支柱7 TASKS.md) remote environment setup -- run this ON THE RENTED
# GPU BOX (Vast.ai / RunPod instance), not locally. See
# notes/cloud-bullet2-execution-plan_2026-08-29.md for the full sequence this
# fits into.
#
# This script installs and downloads; it does not run the benchmark and does
# not terminate the instance. YOU are responsible for destroying the instance
# when done -- nothing in this repo can do that for you.
set -euo pipefail

echo "== disk / GPU sanity =="
df -h /
nvidia-smi

echo "== python env =="
python3 -m venv ~/cloud_bench_venv
source ~/cloud_bench_venv/bin/activate
pip install --upgrade pip

echo "== vllm + guidellm =="
# Pin nothing here on purpose -- check current vLLM V1 EAGLE3 support before
# installing (this repo's local .venv has never had vllm/guidellm installed,
# so these versions were never actually tested against this script).
pip install "vllm>=0.9" guidellm

echo "== CLI syntax check (DO THIS BEFORE RUNNING THE MATRIX) =="
echo "GuideLLM has had two CLI generations (flag-based 'guidellm benchmark'"
echo "vs newer 'guidellm run --backend kind=...'). Confirm which one your"
echo "installed version uses:"
guidellm --help || true
guidellm benchmark --help || true

echo "== HF auth (Llama-3.1 + Llama-3.2-1B are gated / require accepting license + Meta's review) =="
echo "Run: huggingface-cli login   (needs a token with access to meta-llama/Llama-3.1-8B-Instruct"
echo "and meta-llama/Llama-3.2-1B-Instruct -- check both show 'Gated model You have been granted"
echo "access' on huggingface.co BEFORE renting, not 'pending' -- a still-pending request 403s here"
echo "and you've paid for setup time for nothing)."

echo "== prefetch models (avoids paying GPU-idle time for downloads later) =="
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "meta-llama/Llama-3.1-8B-Instruct",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "meta-llama/Llama-3.2-1B-Instruct",
]:
    print("prefetching", repo)
    snapshot_download(repo)
PY

echo "setup done. Next: python -m cloud_bench.orchestrate --dry-run to see the"
echo "planned command matrix, then the sanity check, then --execute."
echo ""
echo "REMINDER: destroy this instance from the Vast.ai/RunPod dashboard when"
echo "the run is done. Nothing here does it automatically."
