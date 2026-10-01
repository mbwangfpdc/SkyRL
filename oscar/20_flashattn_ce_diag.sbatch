#!/bin/bash
# Minimal diagnostic: does the actual container venv used by real training jobs
# successfully import flash_attn.ops.triton.cross_entropy (the fused Triton
# cross-entropy kernel skyrl/backends/skyrl_train/utils/torch_utils.py tries
# first for logprobs_from_logits)? If not, SkyRL silently falls back to
# logprobs_from_logits_v2, which for bf16 logits (our reduce_dtype/autocast
# config) does a per-row, unvectorized Python-loop full log_softmax over the
# entire ~152k-token Qwen vocab -- a real candidate explanation for the
# forward-pass slowdown measured against granular's own manual lm_head +
# gathered-logit + logsumexp approach (granular's fb_trunk_forward/
# fb_head_forward path never materializes a full log_softmax tensor).
#
# No ray/vllm/model loading needed -- this is a plain python import check, so
# it should finish in well under a minute once the job starts.
#
# Branch: cpu-adam-on-repro, with this session's [perf-diag] log line added to
# torch_utils.py's try/except (uncommitted).

#SBATCH --job-name=skyrl-flashattn-ce-diag
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:10:00
#SBATCH --exclude=gpu2709,gpu3006
#SBATCH --output=/oscar/data/deeptir/mborjigi/skyrl/logs/%x-%j.out

set -euo pipefail
source /oscar/data/deeptir/mborjigi/skyrl/env.sh
skyrl_set_jobtmp

SRC="$SKYRL_ROOT/SkyRL"
export SKYRL_GPU=1

echo "node=$(hostname)"
SRC_BRANCH="$(git -C "$SRC" branch --show-current)"
echo "SRC=$SRC  branch=$SRC_BRANCH"
if [ "$SRC_BRANCH" != "cpu-adam-on-repro" ]; then
    echo "FATAL: expected SRC to be on cpu-adam-on-repro, got '$SRC_BRANCH'."
    exit 1
fi
if ! grep -q "perf-diag" "$SRC/skyrl/backends/skyrl_train/utils/torch_utils.py"; then
    echo "FATAL: the [perf-diag] logging addition is not present in torch_utils.py -- refusing to run without it."
    exit 1
fi

skyrl_run bash -c "
    set -euo pipefail
    cd '$SRC'
    $UV run --no-sync --extra fsdp python -c \"
import torch
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
from skyrl.backends.skyrl_train.utils.torch_utils import FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE
print('FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE =', FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE)
import flash_attn
print('flash_attn version:', flash_attn.__version__)
try:
    import triton
    print('triton version:', triton.__version__)
except Exception as e:
    print('triton import FAILED:', type(e).__name__, e)
\"
"
