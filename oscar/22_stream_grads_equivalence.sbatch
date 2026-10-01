#!/bin/bash
# Correctness + memory check for optimizer_config.stream_grads_to_cpu (ZeRO-Offload-style
# per-layer gradient streaming on top of cpu_adam), branch zero-offload-grad-stream.
# ONE test per job, selected by MODE (submit twice):
#   MODE=cpu_adam  old path: cpu_adam, grads accumulate on GPU, one bulk copy after backward
#   MODE=stream    new path: cpu_adam + stream_grads_to_cpu (per-layer D2H during backward)
# Both bf16 for everything except Adam (bf16 weights/compute/reduce-scatter/grads, fp32 CPU
# AdamW masters+moments); same random ~200M Qwen2, same data, 3 steps x 3 micro-batches, clipping
# active. Results -> $OUT/{MODE}_rank{r}.pt; 23_stream_grads_compare.sbatch compares the two.
#
# Self-contained on purpose: /oscar/scratch/mborjigi/skyrl/env.sh and the shared venv's torch
# sources were deleted by the 2026-09-27 scratch purge, so this builds a throwaway venv on
# node-local /tmp from the worktree's uv.lock (no Ray, no Apptainer needed for this check).
# vllm-router is skipped: its only wheel needs glibc 2.35 (host has 2.34) and this check never
# imports it (the full runs get it via the Apptainer image).

#SBATCH --job-name=skyrl-stream-grads-test
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu2709,gpu3006
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out

# Usage: sbatch --export=ALL,MODE=cpu_adam,OUT=<dir>[,SEQ_LEN=4096][,TAG=cpu_adam2] 22_stream_grads_equivalence.sbatch

set -euo pipefail
: "${MODE:?set MODE=cpu_adam|stream}"
: "${OUT:?set OUT=<results dir>}"
SRC=/oscar/scratch/mborjigi/skyrl/SkyRL-zero-offload
TMPD="/tmp/${USER}-stream-grads-${SLURM_JOB_ID}"
mkdir -p "$TMPD"
trap 'rm -rf "$TMPD"' EXIT

echo "MODE=$MODE OUT=$OUT node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-?} branch=$(git -C "$SRC" branch --show-current) head=$(git -C "$SRC" rev-parse --short HEAD)"
grep -q "stream_grads_to_cpu" "$SRC/skyrl/train/config/config.py" || { echo "FATAL: stream_grads_to_cpu missing in $SRC"; exit 1; }

export UV_CACHE_DIR="$TMPD/uv-cache"
export UV_PROJECT_ENVIRONMENT="$TMPD/venv"
export HF_HOME="$TMPD/hf"
export SKYRL_CPU_ADAM_THREADS=2

cd "$SRC"
~/.local/bin/uv sync --frozen --extra fsdp --no-install-package vllm-router
~/.local/bin/uv run --no-sync \
    torchrun --standalone --nproc_per_node=2 \
    tests/backends/skyrl_train/gpu/stream_grads_equivalence_check.py run --mode "$MODE" --out "$OUT" --seq "${SEQ_LEN:-64}" --tag "${TAG:-$MODE}"
