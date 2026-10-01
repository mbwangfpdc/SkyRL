#!/bin/bash
# Forward-pass profiling diagnostic (residual-throughput investigation):
# enables SKYRL_PROFILE_FWD=1 (SKYRL_PROFILE_FWD_EVERY_N=1, so every call is
# profiled) to get model_wrapper.forward's own built-in per-sub-component
# breakdown ([perf-breakdown] model_wrapper.forward: prep=... unpad=...
# hf_model_fwd=... logprobs_from_logits=... repad_logprobs=... entropy=...s),
# logged from workers/model_wrapper.py:416-419.
#
# Context: measured SkyRL's forward pass at ~3.13s/microbatch vs granular's
# ~0.775s/microbatch (~4x), a residual not explained by data volume (already
# fixed separately) or by the flash-attn fused cross-entropy kernel (confirmed
# available and in use via job 6276086's [perf-diag] log line -- ruling out
# the slow logprobs_from_logits_v2 python-loop fallback as the cause).
# Remaining hypothesis: model_wrapper.forward always computes `entropy`
# (compute_entropy=True is passed unconditionally; only entropy_requires_grad
# toggles on self.cfg.algorithm.use_entropy_loss) -- an extra full-vocab
# reduction on every forward call -- whereas granular's compute_token_logprobs
# only computes entropy `want_entropy=(self.cfg.entropy_coef > 0.0)`, skipping
# it entirely when unused. This job's real per-microbatch breakdown will
# either confirm or rule this out directly, rather than guessing further.
#
# Only 1 training step needed (trainer.max_training_steps=1) -- this is a
# profiling smoke test, not a correctness or full-run test.
#
# Branch: cpu-adam-on-repro, same guard pattern as script 19.

#SBATCH --job-name=skyrl-fwd-profile-diag
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768g
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu2709,gpu3006
#SBATCH --output=/oscar/data/deeptir/mborjigi/skyrl/logs/%x-%j.out

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

RUN_NAME=${RUN_NAME:-skyrlsql_fwd_profile_diag}
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
if ! grep -q "PROMPT_STUB_LEN = 1" "$SRC/skyrl/train/trainer.py"; then
    echo "FATAL: the prompt-truncation follow-up fix is not present in $SRC/skyrl/train/trainer.py."
    exit 1
fi
echo "run=$RUN_NAME  ckpt=$CKPT_PATH  export=$EXPORT_PATH  jobtmp=$JOBTMP"

ulimit -u "$(ulimit -Hu)"
echo "nproc limit: $(ulimit -u)"

DB_PATH="$SQL_DB_PATH"

echo "=== starting ray head + SkyRL-SQL (SKYRL_PROFILE_FWD=1, 1-step diagnostic) ==="
skyrl_run bash -c "
    set -euo pipefail
    cd '$SRC'

    ulimit -n \"\$(ulimit -Hn)\"
    echo \"nofile limit (in-container): \$(ulimit -n)\"

    export SKYRL_PROFILE_FWD=1
    export SKYRL_PROFILE_FWD_EVERY_N=2

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
        trainer.max_training_steps=1 \
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
