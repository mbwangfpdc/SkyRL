#!/bin/bash
# Smoke test for the new zero_variance_filter compute-truncation patch (this
# session, uncommitted on cpu-adam-on-repro): Trainer.postprocess_generator_output
# now shrinks zero-variance-group trajectories to a 1-token stub (response_ids/
# loss_masks/rollout_logprobs/rollout_expert_indices all truncated) instead of
# only zeroing loss_mask over the full completion length -- real compute
# savings, not just gradient correctness (see skyrl-cpu-adam-port memory for
# the full "granular-style drop_zero_signal in SkyRL" investigation that led
# here: dynamic_sampling.type="filter" is DAPO's active resampling/backfill,
# NOT what granular's drop_zero_signal does; zero_variance_filter is the real
# analog -- a fixed single batch draw, loss-masked, no backfill).
#
# This is a CORRECTNESS smoke test for the new truncation code path, not a
# full run: does it crash (compute_prompt_mini_batch_boundaries' fixed
# train_batch_size assertion is the main risk -- trajectory *count* must stay
# unchanged even though content is truncated), and does the
# reward/num_zero_variance_filtered metric show groups actually getting
# caught and truncated.
#
# Branch: cpu-adam-on-repro, same guard pattern as script 18. SRC intentionally
# NOT overridden -- the primary checkout is already on this branch, with this
# session's zero_variance_filter truncation patch applied on top (uncommitted).
#
# Deliberately different from script 18 (the real e2e run):
#   - trainer.algorithm.zero_variance_filter=true (+ tol=0.0 for exact
#     std==0 equality, matching granular's own `std_val > 1e-8` check) instead
#     of dynamic_sampling.type=filter -- this is the actual fix, replacing the
#     wrong (DAPO-resampling) mechanism from the e2e run with the real
#     granular-analog mechanism.
#   - trainer.max_training_steps=3: a quick smoke test, not a real run. Safe
#     to use here since dynamic_sampling.type is now unset (None) --
#     handle_dynamic_sampling always returns keep_sampling=False immediately
#     in that case, so every raw batch draw is exactly one real step (no
#     resample-round ambiguity like the earlier dynamic_sampling="filter"
#     step-count confusion this evening).
#   - trainer.resume_mode=none, fresh run_name/ckpt/export paths: standalone
#     smoke test, must not touch the completed skyrlsql_cpu_adam_e2e run's
#     checkpoint directory.
#   - trainer.eval_interval effectively disabled (very high) and
#     eval_before_train=false: eval isn't relevant to validating this patch,
#     skip it to keep this fast.
#   - --time=01:00:00, no requeue-resume machinery needed for a 3-step smoke
#     test.
#   - --exclude=gpu2709,gpu3006: both confirmed bad nodes tonight (ActorDiedError
#     during build_models on gpu2709 twice; uncorrectable ECC error on gpu3006).

#SBATCH --job-name=skyrl-zerovar-smoke
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768g
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu2709,gpu3006
#SBATCH --output=/oscar/data/deeptir/mborjigi/skyrl/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail
source /oscar/data/deeptir/mborjigi/skyrl/env.sh
skyrl_set_jobtmp

SRC="$SKYRL_ROOT/SkyRL"

export SKYRL_GPU=1

NUM_GPUS=4
NUM_INFERENCE_ENGINES=4
TP_SIZE=1
MAX_INPUT_LENGTH=8192
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256

RUN_NAME=${RUN_NAME:-skyrlsql_zerovar_smoke}
LOGGER=${LOGGER:-console}
STAGE_DB=${STAGE_DB:-0}

CKPT_PATH="$CKPT_ROOT/$RUN_NAME"
EXPORT_PATH="$SKYRL_ROOT/exports/$RUN_NAME"

echo "node=$(hostname)  gpus=$SLURM_GPUS_ON_NODE  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE:-?}M"
SRC_BRANCH="$(git -C "$SRC" branch --show-current)"
echo "SRC=$SRC  branch=$SRC_BRANCH"
if [ "$SRC_BRANCH" != "cpu-adam-on-repro" ]; then
    echo "FATAL: expected SRC to be on cpu-adam-on-repro, got '$SRC_BRANCH'. Refusing to run against the wrong branch."
    exit 1
fi
if ! grep -q "zero_variance_filter" "$SRC/skyrl/train/config/config.py"; then
    echo "FATAL: zero_variance_filter not found in $SRC/skyrl/train/config/config.py -- checkout is not what it claims to be."
    exit 1
fi
if ! grep -q "_apply_zero_variance_filter" "$SRC/skyrl/train/trainer.py"; then
    echo "FATAL: the new truncation patch is not present in $SRC/skyrl/train/trainer.py -- refusing to run without it (that's the whole point of this smoke test)."
    exit 1
fi
if ! grep -q "seq_rewards = \[float(sum(r)) for r in rewards\]" "$SRC/skyrl/train/trainer.py"; then
    echo "FATAL: the token-level-reward fix is not present in $SRC/skyrl/train/trainer.py -- refusing to run without it."
    exit 1
fi
if ! grep -q "PROMPT_STUB_LEN = 1" "$SRC/skyrl/train/trainer.py"; then
    echo "FATAL: the prompt-truncation follow-up fix is not present in $SRC/skyrl/train/trainer.py -- refusing to run without it (that's the whole point of this smoke test)."
    exit 1
fi
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

echo "=== starting ray head + SkyRL-SQL (zero_variance_filter smoke test) ==="
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
        trainer.max_training_steps=3 \
        trainer.placement.colocate_all=true \
        trainer.strategy=fsdp \
        trainer.policy.fsdp_config.cpu_offload=false \
        trainer.policy.fsdp_config.mixed_precision.reduce_dtype=bf16 \
        trainer.policy.optimizer_config.cpu_adam=true \
        trainer.ref.fsdp_config.cpu_offload=true \
        trainer.policy.optimizer_config.max_grad_norm=0.5 \
        trainer.policy.sequence_parallel_size=1 \
        trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
        trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
        trainer.algorithm.zero_variance_filter=true \
        trainer.algorithm.zero_variance_filter_tol=0.0 \
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
        trainer.ckpt_interval=1000 \
        trainer.max_ckpts_to_keep=1 \
        trainer.hf_save_interval=1000 \
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
        trainer.ckpt_path='$CKPT_PATH' \
        trainer.export_path='$EXPORT_PATH' \
        trainer.log_path='$LOG_ROOT/$RUN_NAME' \
        trainer.eval_batch_size=32 \
        trainer.eval_before_train=false \
        trainer.eval_interval=1000 \
        ${EXTRA_OVERRIDES:-}
"

echo "job finished (exit=$?)."
