#!/bin/bash
# Functional smoke test for the new optimizer_config.cpu_adam mode
# (skyrl/backends/skyrl_train/distributed/fsdp_strategy.py, branch
# cpu-adam-optimizer, worktree /oscar/scratch/mborjigi/skyrl/SkyRL-cpu-adam,
# commit 10040db9). Not a throughput comparison -- just: does it train
# correctly, and are the per-step overheads where the design says they
# should be (grad_copy/writeback small & PCIe-bound, step_cpu CPU-bound and
# NOT gated on GPU work)?
#
# Otherwise identical to 14_matched_2step_accounting.sbatch (same sibling to
# granular's sql_matched_2step_accounting.script -- see that file's header
# for the STAGE_DB=0 / --mem=768g rationale), except:
#   - SRC points at the cpu-adam-optimizer worktree, not the main checkout.
#   - trainer.policy.fsdp_config.cpu_offload=false (cpu_adam raises if this
#     is true -- see FSDPStrategy.__init__).
#   - trainer.policy.optimizer_config.cpu_adam=true.
#   - offload_after_step deliberately LEFT at its default (true): the
#     implementation forces it inert once cpu_adam=true, so a clean
#     [cpu-adam-breakdown] log (no leftover GPU<->CPU optimizer-state
#     round trip stealing time) is itself evidence that supersession
#     actually works, not just an assumption.
#   - ref keeps fsdp_config.cpu_offload=true, unchanged: ref is
#     inference_only_init (no optimizer), so cpu_adam's incompatibility
#     check never even runs for it.
#
# What to check once it finishes:
#   - "[cpu_adam] rank=... torch intra-op CPU threads -> N" once per policy
#     rank at init (confirms the mode was actually taken, not silently a
#     no-op).
#   - "[cpu-adam-breakdown] rank=...: grad_copy_gpu2cpu=...s step_cpu=...s
#     writeback_cpu2gpu=...s total=...s" once per policy rank per optimizer
#     step (2 steps x however many policy ranks). grad_copy/writeback should
#     each be a small fraction of a second (one-shot grad/weight-sized PCIE
#     copy); step_cpu should dominate the total and be roughly constant
#     step-to-step (CPU AdamW work, independent of anything on GPU).
#   - Step 1's loss / policy grad-norm should match job 5185074's step 1
#     (skyrlsql_matched_2step_accounting, cpu_offload=true, same data order
#     and same initial weights) almost exactly -- cpu_adam only changes
#     optimizer.step(), not forward/backward, so step 1 (before any weight
#     update has happened) is a same-inputs cross-check for free. Small
#     divergence from step 2 onward (CPU foreach AdamW vs GPU fused AdamW
#     last-ULP differences propagating through the weight update) is
#     expected and fine.
#   - No CUDA errors, no crash, exit 0 (or a clean crash-during-shutdown
#     after both steps have already logged their full Step summary, same
#     pattern documented for job 5182650/5185074).
#
# NOT covered by this run (deliberately out of scope for a first smoke
# test -- see the cpu-adam-optimizer commit message and the granular-vs-
# skyrl-throughput-investigation memory for the follow-ups): a same-world-
# size save->resume cycle confirming the reloaded Adam moments are correct,
# and a throughput comparison against cpu_offload=true / offload_after_step.

#SBATCH --job-name=skyrl-cpu-adam-smoketest
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768g
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail
source /oscar/scratch/mborjigi/skyrl/env.sh
skyrl_set_jobtmp

# Test the cpu-adam-optimizer worktree's code, not the main checkout.
SRC="$SKYRL_ROOT/SkyRL-cpu-adam"

export SKYRL_GPU=1

NUM_GPUS=4
NUM_INFERENCE_ENGINES=4
TP_SIZE=1
MAX_INPUT_LENGTH=8192
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256

RUN_NAME=${RUN_NAME:-skyrlsql_cpu_adam_smoketest}
LOGGER=${LOGGER:-console}
STAGE_DB=${STAGE_DB:-0}

CKPT_PATH="$CKPT_ROOT/$RUN_NAME"
EXPORT_PATH="$SKYRL_ROOT/exports/$RUN_NAME"

echo "node=$(hostname)  gpus=$SLURM_GPUS_ON_NODE  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "SRC=$SRC"
echo "run=$RUN_NAME  ckpt=$CKPT_PATH  export=$EXPORT_PATH  jobtmp=$JOBTMP"

ulimit -u "$(ulimit -Hu)"
echo "nproc limit: $(ulimit -u)"

DB_PATH="$SQL_DB_PATH"
if [ "$STAGE_DB" = "1" ]; then
    STAGED="$JOBTMP/db/data"
    echo "=== staging SynSQL-2.5M + spider to $STAGED ==="
    mkdir -p "$STAGED"
    time cp -r "$SQL_DB_PATH/SynSQL-2.5M" "$SQL_DB_PATH/spider" "$STAGED/"
    DB_PATH="$STAGED"
    du -sh "$STAGED"
fi
echo "DB_PATH=$DB_PATH"

echo "=== scratch usage before start ==="
checkquota 2>/dev/null | awk '/oscar.scratch/ {print}' || df -h /oscar/scratch/mborjigi

echo "=== starting ray head + SkyRL-SQL (cpu_adam=true) ==="
skyrl_run bash -c "
    set -euo pipefail
    cd '$SRC'

    ulimit -n \"\$(ulimit -Hn)\"
    echo \"nofile limit (in-container): \$(ulimit -n)\"

    cleanup() { $UV run --no-sync --extra fsdp ray stop --force >/dev/null 2>&1 || true; }
    trap cleanup EXIT

    $UV run --no-sync --extra fsdp ray start --head \
        --port=6379 \
        --temp-dir='$JOBTMP/ray' \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$NUM_GPUS \
        --disable-usage-stats

    export RAY_ADDRESS='127.0.0.1:6379'

    $UV run --no-sync --extra fsdp -m skyrl.train.entrypoints.main_base \
        trainer.algorithm.advantage_estimator='grpo' \
        data.train_data=\"['$SQL_DATA/train.parquet']\" \
        data.val_data=\"['$SQL_DATA/validation.parquet']\" \
        trainer.policy.model.path='Qwen/Qwen2.5-Coder-7B-Instruct' \
        trainer.epochs=18 \
        trainer.placement.colocate_all=true \
        trainer.strategy=fsdp \
        trainer.policy.fsdp_config.cpu_offload=false \
        trainer.policy.optimizer_config.cpu_adam=true \
        trainer.ref.fsdp_config.cpu_offload=true \
        trainer.policy.optimizer_config.max_grad_norm=0.5 \
        trainer.policy.sequence_parallel_size=1 \
        trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
        trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
        generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
        generator.inference_engine.tensor_parallel_size=$TP_SIZE \
        trainer.train_batch_size=$TRAIN_BATCH_SIZE \
        trainer.max_tokens_per_microbatch=28000 \
        trainer.max_prompt_length=4096 \
        generator.max_input_length=$MAX_INPUT_LENGTH \
        generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy_mini_batch_size=256 \
        trainer.algorithm.use_kl_loss=false \
        trainer.ckpt_interval=10 \
        trainer.max_ckpts_to_keep=1 \
        trainer.hf_save_interval=30 \
        trainer.dump_data_batch=false \
        generator.inference_engine.backend=vllm \
        generator.inference_engine.run_engines_locally=true \
        generator.inference_engine.weight_sync_backend=nccl \
        generator.batched=false \
        environment.env_class=text2sql \
        generator.use_conversation_multi_turn=false \
        generator.n_samples_per_prompt=5 \
        generator.inference_engine.gpu_memory_utilization=0.9 \
        generator.inference_engine.max_num_seqs=2048 \
        generator.inference_engine.enforce_eager=true \
        generator.max_turns=5 \
        generator.sampling_params.temperature=0.6 \
        generator.sampling_params.top_p=0.95 \
        generator.sampling_params.stop='[\"</sql>\", \"</solution>\"]' \
        generator.eval_sampling_params.stop='[\"</sql>\", \"</solution>\"]' \
        generator.eval_sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
        environment.skyrl_gym.text2sql.db_path='$DB_PATH' \
        trainer.logger='$LOGGER' \
        trainer.project_name='skyrlsql' \
        trainer.run_name='$RUN_NAME' \
        trainer.resume_mode=none \
        trainer.max_training_steps=2 \
        trainer.ckpt_path='$CKPT_PATH' \
        trainer.export_path='$EXPORT_PATH' \
        trainer.log_path='$LOG_ROOT/$RUN_NAME' \
        trainer.eval_batch_size=32 \
        trainer.eval_before_train=false \
        trainer.eval_interval=1000 \
        ${EXTRA_OVERRIDES:-}
" &
TRAIN_PID=$!
echo "training PID=$TRAIN_PID"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 300
    skyrl_prune_exports "$EXPORT_PATH" 2
done

wait "$TRAIN_PID"
TRAIN_EXIT=$?

echo "=== pruning to final checkpoint only ==="
skyrl_prune_exports "$EXPORT_PATH" 1
ls -la "$EXPORT_PATH" 2>/dev/null || true

echo "=== scratch usage after finish ==="
checkquota 2>/dev/null | awk '/oscar.scratch/ {print}' || df -h /oscar/scratch/mborjigi

echo "job finished (exit=$TRAIN_EXIT)."
exit "$TRAIN_EXIT"
