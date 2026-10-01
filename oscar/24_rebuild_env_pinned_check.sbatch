#!/bin/bash
# Rebuild the SkyRL Apptainer environment after the 2026-09-27 scratch purge, then run the
# pinned_host_stage correctness/timing check (branch cpu-adam-pinned-stage, worktree SkyRL-pinned).
#
# The purge deleted the SIF, env.sh (restored from session history; backup copy in
# /oscar/data/deeptir/mborjigi/skyrl_backup/) and the venv's torch .py sources. The uv cache shares
# inodes (hardlink mode) with the venv, so it is treated as damaged too: both are deleted and
# rebuilt from uv.lock. The venv is synced from SkyRL-pinned, so its editable skyrl install points
# there (runs set SRC to the same tree).
#
#   sbatch 24_rebuild_env_pinned_check.sbatch

#SBATCH --job-name=skyrl-rebuild-pinned-check
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu2709,gpu3006,gpu3005
#SBATCH --output=/oscar/data/deeptir/mborjigi/skyrl/logs/%x-%j.out

set -euo pipefail
source /oscar/data/deeptir/mborjigi/skyrl/env.sh
skyrl_set_jobtmp
SRC="$SKYRL_ROOT/SkyRL-pinned"
echo "node=$(hostname) src=$SRC branch=$(git -C "$SRC" branch --show-current) head=$(git -C "$SRC" rev-parse --short HEAD)"
grep -q "pinned_host_stage" "$SRC/skyrl/train/config/config.py" || { echo "FATAL: pinned_host_stage missing"; exit 1; }

skyrl_check_egress

export APPTAINER_CACHEDIR="$JOBTMP/apptainer-cache" APPTAINER_TMPDIR="$JOBTMP/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
if [ ! -s "$SIF" ]; then
    echo "=== pulling $SKYRL_IMAGE_URI -> $SIF ==="
    time apptainer pull --force "$SIF.partial" "$SKYRL_IMAGE_URI"
    mv "$SIF.partial" "$SIF"
fi

echo "=== deleting purge-damaged venv + uv cache ==="
time rm -rf "$VENV" "$UV_CACHE"

export SKYRL_GPU=1
skyrl_run bash -c "
    set -euo pipefail
    cd '$SRC'
    echo '=== uv sync ==='
    time $UV sync --frozen --extra fsdp
    $UV run --no-sync python -c 'import torch, vllm, ray, flash_attn, skyrl; print(\"IMPORT_OK torch\", torch.__version__, \"vllm\", vllm.__version__, \"ray\", ray.__version__, \"skyrl\", skyrl.__file__)'
    echo '=== pinned_stage_check ==='
    $UV run --no-sync torchrun --standalone --nproc_per_node=2 tests/backends/skyrl_train/gpu/pinned_stage_check.py
"
