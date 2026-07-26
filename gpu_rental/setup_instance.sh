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

# Refuse to run against an existing venv (2026-07-27, found the hard way):
# the corrupted-torch incident this script was rewritten to prevent came
# from running the manual torch pin (below) repeatedly against a venv that
# already had SOME torch installed (by a previous partial run, or by `uv
# run`'s auto-sync before UV_NO_SYNC was in place) -- each reinstall left
# behind stale dist-info/compiled-extension remnants from the prior
# version, and torch's inductor compiler (which reads its own source
# layout at import time) ended up reading a mix of two versions' files.
# `--force-reinstall` below helps but a from-scratch venv is the only
# actually-verified-clean starting point. Fail fast and say so rather than
# silently installing on top of unknown existing state.
if [ -d .venv ]; then
  echo "FATAL: .venv already exists. This script must run against a fresh venv --" >&2
  echo "installing on top of an existing one is exactly how the mixed-torch-version" >&2
  echo "corruption happened before. Run 'rm -rf .venv' first, then rerun this script." >&2
  exit 1
fi

# Persist HF_HOME + UV_CACHE_DIR + UV_NO_SYNC to every future shell on this
# box (new tmux windows, reconnected SSH sessions) -- not just this script's
# own environment.
#
# HF_HOME/UV_CACHE_DIR: without these, huggingface-cli/uv silently fall back
# to ~/.cache/{huggingface,uv} on RunPod's Container Disk (60GB, tied to pod
# lifecycle), undoing the /workspace (Network Volume, 180GB) redirect the
# rest of this repo's paths depend on (see RUNBOOK.md's disk-location rule)
# -- a 23GB checkpoint landing there mid-download is exactly the "found out
# hours into an unattended overnight run" failure this is meant to prevent.
#
# UV_NO_SYNC=1 (2026-07-26, found the hard way): `uv run` defaults to
# re-syncing the venv against uv.lock before running anything -- and
# uv.lock resolves torch to a generic PyPI version (2.13.0 as of this
# writing), NOT the exact-pinned torch==2.11.0+cu129 installed manually two
# steps below via `uv pip install` (which bypasses the lock on purpose, see
# that install's own comment). Without this, EVERY `uv run` call --
# including this very script's own version-check three lines down, and
# every later `uv run python -m scripts.precompute_llm_annotations ...`
# invocation in RUNBOOK.md -- silently uninstalls the pinned cu129 torch and
# reinstalls the generic one first, undoing the pin on every single
# invocation. `--no-sync` is the equivalent per-call flag if you ever need
# to override this for one command; the env var is set here so nothing
# downstream has to remember to add it.
#
# Written to both ~/.bashrc (new tmux windows: interactive non-login shells
# read this) and /etc/profile.d (SSH login shells) for coverage; exported
# directly below too so THIS script's own remaining uv/hf calls already use
# it. Idempotent -- grep guards against duplicate lines on repeated runs.
ENV_BLOCK='export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_NO_SYNC=1'
if ! grep -qF "HF_HOME=/workspace/hf" "$HOME/.bashrc" 2>/dev/null; then
  printf '\n%s\n' "$ENV_BLOCK" >> "$HOME/.bashrc"
fi
if [ -w /etc/profile.d ] && [ ! -f /etc/profile.d/trace-the-ace-env.sh ]; then
  printf '%s\n' "$ENV_BLOCK" > /etc/profile.d/trace-the-ace-env.sh
fi
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_NO_SYNC=1
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# --no-install-package torch: pyproject.toml lists an unpinned "torch" (needed
# transitively by sentence-transformers regardless of that explicit line, so
# removing the line wouldn't help) -- without this flag, uv would install some
# generic PyPI torch build here (with its own multi-GB NVIDIA CUDA dependency
# set) only to have it immediately overwritten by the exact-pinned cu129
# wheel below. This flag still resolves torch normally (satisfies
# sentence-transformers' constraint in the lock) but skips physically
# installing/downloading it, so it only gets fetched once, in the pinned
# form the competition container actually runs. torchvision/torchaudio
# aren't in THIS project's pyproject.toml/uv.lock at all (grepped, absent --
# nothing in src/ imports either one, this is a text-only pipeline), so
# there's nothing to exclude for them here; they only enter the picture
# via the vllm wheel below.
uv sync --no-install-package torch
echo "=== .venv size (should land under /workspace since cwd is the cloned repo there) ==="
du -sh .venv

# vllm FIRST, torch/torchvision/torchaudio pin LAST -- reordered from the
# previous torch-then-vllm sequence (2026-07-27, root-caused the
# corruption incident). The competition runtime
# (tutoring-outcomes-runtime/runtime/pyproject.toml) pins all three
# together from the SAME pytorch-cu129 index (torch==2.11.0+cu129,
# torchvision==0.26.0+cu129, torchaudio==2.11.0+cu129) -- torchvision in
# particular is never mentioned in trace-the-ace's own dependencies, so it
# only entered this venv at all via the vllm wheel's own transitive
# requirement, using whatever unpinned/generic-index version pip's
# resolver picked for it. Installing torch alone first (the old order)
# left a window where vllm's own install could still drag in a mismatched
# torchvision (or nudge torch itself) to satisfy ITS resolution -- e.g. a
# generic torchvision 0.26.0 built against generic-PyPI torch 2.13.0
# landing next to our pinned cu129 torch, exactly the
# ModuleNotFoundError/mixed-dist-info state hit in practice. Installing
# vllm first (let it pull in whatever transitive torch stack it wants),
# then force-reinstalling the exact matching cu129 triplet from PyTorch's
# own index as ONE atomic resolution, guarantees the pinned triplet is the
# last thing written and that torch/torchvision/torchaudio are mutually
# consistent (resolved together, not three independent installs) --
# matching the container byte-for-byte the way this script's header
# already claims to. If tutoring-outcomes-runtime's pins change, re-copy
# the current values from there rather than trusting these.
uv pip install "https://wheels.vllm.ai/ad7125a431e176d4161099480a66f0169609a690/vllm-0.21.0%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl"
uv pip install --force-reinstall \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129

# Self-check: fail loudly HERE, before any multi-GB model download, rather
# than discovering a broken torch/vllm pairing hours into an unattended
# run (the inductor-compiler-crash failure mode this whole rewrite exists
# to catch earlier). Checks, in order: (1) exact pinned versions actually
# landed -- catches force-reinstall silently resolving to something else;
# (2) CUDA actually visible to torch; (3) transformers imports clean
# (exercises the torch/torchvision pairing transformers' own import graph
# touches); (4) vllm imports AND its LLM class is constructible-import
# clean (exercises vllm's own compiled-extension binding against the
# force-reinstalled torch -- this is the one step that would have caught
# the incident immediately instead of hours later).
echo "=== self-check: pinned versions + imports ==="
uv run python -c "
from importlib.metadata import version

expected = {
    'torch': '2.11.0+cu129',
    'torchvision': '0.26.0+cu129',
    'torchaudio': '2.11.0+cu129',
}
for pkg, want in expected.items():
    got = version(pkg)
    assert got == want, f'{pkg} version mismatch: expected {want!r}, got {got!r}'

import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'

import transformers
import vllm
from vllm import LLM

print('self-check passed:')
print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('  torchvision', version('torchvision'))
print('  torchaudio', version('torchaudio'))
print('  transformers', transformers.__version__)
print('  vllm', vllm.__version__)
" || { echo "FATAL: environment self-check failed (see error above). Do NOT proceed to model downloads -- fix this first." >&2; exit 1; }

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# hf_transfer speeds up the multi-GB checkpoint downloads materially --
# install it into the same venv, no need to add it to pyproject.toml (this
# script is the only thing that needs it, and only on the rented box).
uv pip install hf_transfer huggingface_hub

echo ""
echo "=== disk usage after setup (/workspace should hold everything large; / should stay small) ==="
df -h /workspace /

echo ""
echo "Setup done. HF_HOME=/workspace/hf, UV_CACHE_DIR=/workspace/.uv-cache, and UV_NO_SYNC=1 are"
echo "exported inside THIS SCRIPT's own process (a child of your shell) and persisted to ~/.bashrc"
echo "+ /etc/profile.d for future shells/tmux windows -- but NOT retroactively into the shell you"
echo "launched this script FROM (child-process exports never propagate back to the parent shell)."
echo ""
echo ">>> Run 'source ~/.bashrc' in THIS terminal now, before typing any download command below <<<"
echo "(a brand new tmux window/pane would pick these up automatically instead, if you prefer that)."
echo ""
echo "Next: download the candidate model(s), e.g.:"
echo '  HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir ./models/Qwen3-8B-AWQ'
echo '  HF_HUB_ENABLE_HF_TRANSFER=1 uv run huggingface-cli download QuixiAI/Qwen3-30B-A3B-AWQ --local-dir ./models/Qwen3-30B-A3B-AWQ'
echo "See RUNBOOK.md for the rest of the day's sequence (pinned checkpoint choice + backup there)."
