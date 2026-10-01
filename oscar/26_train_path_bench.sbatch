#!/bin/bash
# Why is SkyRL's training forward/backward ~20%/token slower than granular's (job 6828066 vs
# atc-exp18-record)? Same packed microbatches through each path on one node; see
# bench/train_path_bench.py. Variants (each a fresh 4-GPU torchrun):
#   skyrl            SkyRL as run: fp32 load, entropy on          (SkyRL venv, container)
#   skyrl-bf16       + bf16 load
#   skyrl-noent      + entropy off
#   skyrl-bf16-noent both
#   granular         granular's path in the SkyRL venv             (same torch/transformers)
#   granular-own     granular's path in granular's venv            (its own torch 2.10 / tf 4.57)
#
#   sbatch 26_train_path_bench.sbatch

#SBATCH --job-name=skyrl-train-path-bench
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400g
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu2709,gpu3006,gpu3005
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out

set -uo pipefail
source /oscar/scratch/mborjigi/skyrl/env.sh
skyrl_set_jobtmp
SRC="$SKYRL_ROOT/SkyRL-pinned"
BENCH="$SKYRL_ROOT/bench/train_path_bench.py"
GRANULAR=/oscar/scratch/mborjigi/granular-cais-rl/.claude/worktrees/adaptive-threshold-policy
echo "node=$(hostname) skyrl=$(git -C "$SRC" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader | head -1

export SKYRL_GPU=1
run_sky() {  # $1 = tag, rest = bench flags
    local tag=$1; shift
    echo "=== $tag ==="
    skyrl_run bash -c "cd '$SRC' && $UV run --no-sync torchrun --standalone --nproc_per_node=4 '$BENCH' --tag=-$tag $*" 2>&1 \
        | grep -E "BENCH|Error|error|Traceback|OutOfMemory" | head -20
}
run_sky skyrl            --mode skyrl
run_sky skyrl-bf16       --mode skyrl --bf16-load
run_sky skyrl-noent      --mode skyrl --no-entropy
run_sky skyrl-bf16-noent --mode skyrl --bf16-load --no-entropy
run_sky granular         --mode granular

echo "=== granular-own ==="
(cd "$GRANULAR" && source .venv/bin/activate && torchrun --standalone --nproc_per_node=4 "$BENCH" --mode granular --tag=-own) 2>&1 \
    | grep -E "BENCH|Error|error|Traceback|OutOfMemory" | head -20
