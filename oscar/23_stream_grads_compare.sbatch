#!/bin/bash
# Compares the two 22_stream_grads_equivalence.sbatch results (MODE=cpu_adam vs MODE=stream) in
# $OUT and prints per-check ok/FAIL plus STREAM_GRADS_EQUIVALENCE: PASS|FAIL. CPU-only; submit
# with --dependency=afterok:<cpu_adam job>:<stream job>.
# 64g: node-local /tmp is RAM-backed and the uv venv (torch+CUDA) alone OOMs 16g (job 6793008).
# Usage: sbatch --export=ALL,OUT=<dir>[,FLOOR=cpu_adam2] --dependency=afterok:A:B 23_stream_grads_compare.sbatch

#SBATCH --job-name=skyrl-stream-grads-compare
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out

set -euo pipefail
: "${OUT:?set OUT=<results dir>}"
SRC=/oscar/scratch/mborjigi/skyrl/SkyRL-zero-offload
TMPD="/tmp/${USER}-stream-grads-cmp-${SLURM_JOB_ID}"
mkdir -p "$TMPD"
trap 'rm -rf "$TMPD"' EXIT
export UV_CACHE_DIR="$TMPD/uv-cache"
export UV_PROJECT_ENVIRONMENT="$TMPD/venv"
cd "$SRC"
echo "OUT=$OUT head=$(git -C "$SRC" rev-parse --short HEAD)"
~/.local/bin/uv sync --frozen --extra fsdp --no-install-package vllm-router
~/.local/bin/uv run --no-sync python tests/backends/skyrl_train/gpu/stream_grads_equivalence_check.py compare --out "$OUT" ${FLOOR:+--floor "$FLOOR"}
