#!/bin/bash
# Empirical probe: what max_tokens_per_microbatch fits on 2 L40S GPUs with
# optimizer_config.cpu_adam=true (7B model, colocate_all=true)? No existing
# validated number covers this -- everything so far (28000) was measured at
# NUM_GPUS=4. Halving GPU count under colocate_all squeezes BOTH sides at
# once: each GPU now holds a larger FSDP shard (halved sharding), AND the
# same aggregate rollout concurrency (train_batch_size=256, n_samples=5) has
# to fit its KV cache across half as many vLLM engines at the same
# gpu_memory_utilization=0.9. A naive linear halving of 28000 (-> ~14000)
# isn't trustworthy given that compounding effect, so this starts more
# conservative (16000) and should be adjusted based on the actual result.
#
# Derived from 15_matched_2step_cpu_adam.sbatch. Diffs:
#   - --gres=gpu:2 (was 4); NUM_GPUS/NUM_INFERENCE_ENGINES/policy_num_gpus_
#     per_node/ref_num_gpus_per_node all =2 (was 4).
#   - SRC points at the track-mbwang checkout (cpu_adam is now merged there,
#     see skyrl-cpu-adam-port memory) instead of the old cpu-adam-optimizer
#     dev worktree.
#   - trainer.max_tokens_per_microbatch=16000 (was 28000) -- the value under
#     test.
#   - trainer.eval_interval=-1 (was 1000) to actually skip eval -- 1000
#     still triggered a mandatory final-step eval in the 4-GPU smoke test
#     (job 5743785, ~11.5 min), which is irrelevant to a memory-ceiling
#     question and just costs wall time here.
#
# What to check: does training complete both steps without a CUDA OOM
# during forward/backward, the cpu_adam grad-copy/writeback, or (per the
# no-offload OOM finding earlier this investigation) weight-sync's
# extract_weights full-tensor gather right after each optimizer step? A
# clean exit (or the established benign shutdown-crash pattern -- Executor
# failed / EngineDeadError AFTER both steps' Step summary already logged)
# means 16000 is safe; a genuine mid-step CUDA OOM means it needs to come
# down.

#SBATCH --job-name=skyrl-2gpu-tokbudget-probe
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768g
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/data/deeptir/mborjigi/skyrl/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail
source /oscar/data/deeptir/mborjigi/skyrl/env.sh
skyrl_set_jobtmp

# track-mbwang: cpu_adam merged there (see skyrl-cpu-adam-port memory) --
# this checkout's own `main` still tracks upstream and lacks cpu_adam.
SRC="$SKYRL_ROOT/SkyRL"

export SKYRL_GPU=1

NUM_GPUS=2
NUM_INFERENCE_ENGINES=2
TP_SIZE=1
MAX_INPUT_LENGTH=8192
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256
TOKEN_BUDGET=${TOKEN_BUDGET:-16000}

RUN_NAME=${RUN_NAME:-skyrlsql_2gpu_tokbudget_probe}
LOGGER=${LOGGER:-console}
STAGE_DB=${STAGE_DB:-0}

CKPT_PATH="$CKPT_ROOT/$RUN_NAME"
EXPORT_PATH="$SKYRL_ROOT/exports/$RUN_NAME"

echo "node=$(hostname)  gpus=$SLURM_GPUS_ON_NODE  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE:-?}M"
SRC_BRANCH="$(git -C "$SRC" branch --show-current)"
echo "SRC=$SRC  branch=$SRC_BRANCH"
if [ "$SRC_BRANCH" != "cpu-adam-on-repro" ]; then
    echo "FATAL: expected SRC to be on cpu-adam-on-repro (cpu_adam rebased onto repro's base," \
         "10d92b8e -- torch==2.11.0/vllm==0.23.0/ray==2.56.0, matches granular AND has" \
         "max_tokens_per_microbatch; track-mbwang turned out to be a sibling branch that never" \
         "merged in max_tokens_per_microbatch at all -- see chat history), got '$SRC_BRANCH'." \
         "Refusing to run against the wrong branch."
    exit 1
fi
if ! grep -q "cpu_adam" "$SRC/skyrl/train/config/config.py"; then
    echo "FATAL: cpu_adam not found in $SRC/skyrl/train/config/config.py -- checkout is not what it claims to be."
    exit 1
fi
if ! grep -q "max_tokens_per_microbatch" "$SRC/skyrl/train/config/config.py"; then
    echo "FATAL: max_tokens_per_microbatch not found in $SRC/skyrl/train/config/config.py -- checkout is not what it claims to be."
    exit 1
fi
echo "run=$RUN_NAME  ckpt=$CKPT_PATH  export=$EXPORT_PATH  jobtmp=$JOBTMP  token_budget=$TOKEN_BUDGET"

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

echo "=== starting ray head + SkyRL-SQL (cpu_adam=true, 2 GPUs, token_budget=$TOKEN_BUDGET) ==="
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
        trainer.max_tokens_per_microbatch=$TOKEN_BUDGET \
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
        trainer.eval_interval=-1 \
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
