#!/usr/bin/env bash
# Run on the rented GPU box, after rsyncing this repo over (see RUNBOOK.md
# step 1). Installs uv + base deps, then vLLM/torch pinned to EXACTLY the
# competition container's versions (tutoring-outcomes-runtime/runtime/
# pyproject.toml) -- not "some vLLM version", the specific wheel the
# competition will run at scoring time, so this rental doubles as a real
# test of that never-yet-run code path.
#
# Usage: bash gpu_rental/setup_instance.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || true
fi

uv sync

# torch + vllm: exact pins/wheel URL copied from
# tutoring-outcomes-runtime/runtime/pyproject.toml as of 2026-07-24. If that
# file has changed since, re-copy the current values rather than trusting
# these -- the whole point is byte-for-byte match with the container.
uv pip install torch==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129
uv pip install "https://wheels.vllm.ai/ad7125a431e176d4161099480a66f0169609a690/vllm-0.21.0%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl"

echo "=== installed versions ==="
uv run python -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__); print('cuda available', torch.cuda.is_available())"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# hf_transfer speeds up the multi-GB checkpoint downloads materially --
# install it into the same venv, no need to add it to pyproject.toml (this
# script is the only thing that needs it, and only on the rented box).
uv pip install hf_transfer huggingface_hub

echo ""
echo "Setup done. Next: download the candidate model(s), e.g.:"
echo '  HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir ./models/Qwen3-8B-AWQ'
echo '  HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download <32B-AWQ-repo> --local-dir ./models/Qwen3-32B-AWQ'
echo "See RUNBOOK.md for the rest of the day's sequence."
