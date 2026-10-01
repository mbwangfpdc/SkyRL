#!/bin/bash
# DPBAL + NUM_WORKERS=0 (in-process prompt loading, no per-epoch worker respawn), with the stall
# instrumentation of 29 (stack sampler now covers all threads). Base: 28_matched_exp18_dpbal_short.sbatch + [stall-debug] timestamps (driver/worker),
# in-process stack sampling (SKYRL_STACK_SAMPLE_DIR) and bench/host_sampler.sh (cgroup memory,
# PSI, GPU util, top procs) -> logs/stall-<jobid>/.
# SHORT (5-step) copy of 25_matched_exp18_cpu_adam_pinned.sbatch with trainer.dp_token_balance=true
# (branch dp-token-balance): each policy mini-batch is reordered so the per-DP-rank slices carry
# equal token totals. Compare training fwd/bwd time per token, microbatches/rank and padding
# microbatches against job 6828066 (same config, stock contiguous split). Runs
# bench/dp_balance_test.py (CPU check of the reorder) before training.
#
#   sbatch 28_matched_exp18_dpbal_short.sbatch

#SBATCH --job-name=skyrl-dpbal-nw0
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512g
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu2709,gpu3006,gpu3005
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail
source /oscar/scratch/mborjigi/skyrl/env.sh
skyrl_set_jobtmp

SRC="$SKYRL_ROOT/SkyRL-pinned"
export SKYRL_GPU=1
NUM_GPUS=4
RUN_NAME=${RUN_NAME:-skyrlsql_dpbal_nw0}
CKPT_PATH="$CKPT_ROOT/$RUN_NAME"
EXPORT_PATH="$SKYRL_ROOT/exports/$RUN_NAME"

echo "node=$(hostname)  gpus=$SLURM_GPUS_ON_NODE  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE:-?}M"
SRC_BRANCH="$(git -C "$SRC" branch --show-current)"
echo "SRC=$SRC  branch=$SRC_BRANCH  head=$(git -C "$SRC" rev-parse --short HEAD)"
[ "$SRC_BRANCH" = "dp-token-balance" ] || { echo "FATAL: expected dp-token-balance, got '$SRC_BRANCH'"; exit 1; }
grep -q "pinned_host_stage" "$SRC/skyrl/train/config/config.py" || { echo "FATAL: pinned_host_stage missing"; exit 1; }
grep -q "_apply_zero_variance_filter" "$SRC/skyrl/train/trainer.py" || { echo "FATAL: zero_variance_filter patch missing"; exit 1; }
[ -s "$SIF" ] || { echo "FATAL: $SIF missing -- run 24_rebuild_env_pinned_check.sbatch first"; exit 1; }

ulimit -u "$(ulimit -Hu)"

export SKYRL_STACK_SAMPLE_DIR="$LOG_ROOT/stall-$SLURM_JOB_ID"
mkdir -p "$SKYRL_STACK_SAMPLE_DIR"
bash "$SKYRL_ROOT/bench/host_sampler.sh" "$SKYRL_STACK_SAMPLE_DIR" 5 &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null; rm -rf "$JOBTMP"' EXIT
echo "stall debug output: $SKYRL_STACK_SAMPLE_DIR"

skyrl_run bash -c "
    set -euo pipefail
    cd '$SRC'
    ulimit -n \"\$(ulimit -Hn)\"
    cleanup() { $UV run --no-sync --extra fsdp ray stop --force >/dev/null 2>&1 || true; }
    trap cleanup EXIT

    $UV run --no-sync --extra fsdp ray start --head \
        --port=6379 \
        --temp-dir='$JOBTMP/ray' \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$NUM_GPUS \
        --disable-usage-stats
    export RAY_ADDRESS='127.0.0.1:6379'

    $UV run --no-sync python $SKYRL_ROOT/bench/dp_balance_test.py 2>&1 | grep -v Warning | tail -6
    $UV run --no-sync python $SKYRL_ROOT/bench/dp_balance_test.py >/dev/null 2>&1 || { echo 'FATAL: dp_balance_test failed'; exit 1; }

    $UV run --no-sync --extra fsdp -m skyrl.train.entrypoints.main_base \
        trainer.algorithm.advantage_estimator='grpo' \
        data.train_data=\"['$SQL_DATA/train.parquet']\" \
        data.val_data=\"['$SQL_DATA/validation.parquet']\" \
        trainer.policy.model.path='Qwen/Qwen2.5-Coder-7B-Instruct' \
        trainer.seed=43 \
        trainer.epochs=18 \
        trainer.max_training_steps=5 \
        trainer.placement.colocate_all=true \
        trainer.strategy=fsdp \
        trainer.policy.fsdp_config.cpu_offload=false \
        trainer.policy.fsdp_config.pinned_host_stage=true \
        trainer.dp_token_balance=true \
        data.dataloader.num_workers=0 \
        trainer.policy.fsdp_config.mixed_precision.param_dtype=bf16 \
        trainer.policy.fsdp_config.mixed_precision.reduce_dtype=fp32 \
        trainer.policy.optimizer_config.cpu_adam=true \
        trainer.policy.optimizer_config.master_dtype=fp32 \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy.optimizer_config.weight_decay=0.01 \
        trainer.policy.optimizer_config.max_grad_norm=0.5 \
        trainer.policy.optimizer_config.num_warmup_steps=0 \
        trainer.policy.optimizer_config.scheduler=constant_with_warmup \
        trainer.policy.sequence_parallel_size=1 \
        trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
        trainer.algorithm.use_kl_loss=false \
        trainer.algorithm.zero_variance_filter=true \
        trainer.algorithm.zero_variance_filter_tol=0.0 \
        trainer.train_batch_size=256 \
        trainer.policy_mini_batch_size=256 \
        trainer.max_tokens_per_microbatch=24000 \
        trainer.max_prompt_length=4096 \
        generator.n_samples_per_prompt=5 \
        generator.max_input_length=24000 \
        generator.max_turns=5 \
        generator.sampling_params.max_generate_length=3000 \
        generator.sampling_params.temperature=0.6 \
        generator.sampling_params.top_p=0.95 \
        generator.sampling_params.stop='[\"</sql>\", \"</solution>\"]' \
        generator.eval_sampling_params.stop='[\"</sql>\", \"</solution>\"]' \
        generator.eval_sampling_params.max_generate_length=3000 \
        generator.inference_engine.num_engines=$NUM_GPUS \
        generator.inference_engine.tensor_parallel_size=1 \
        generator.inference_engine.backend=vllm \
        generator.inference_engine.run_engines_locally=true \
        generator.inference_engine.weight_sync_backend=nccl \
        generator.inference_engine.gpu_memory_utilization=0.9 \
        generator.inference_engine.max_num_seqs=2048 \
        generator.inference_engine.max_num_batched_tokens=8192 \
        generator.inference_engine.enforce_eager=false \
        generator.batched=false \
        generator.use_conversation_multi_turn=false \
        environment.env_class=text2sql \
        environment.skyrl_gym.max_env_workers=2048 \
        environment.skyrl_gym.text2sql.db_path='$SQL_DB_PATH' \
        trainer.ckpt_interval=-1 \
        trainer.hf_save_interval=-1 \
        trainer.resume_mode=none \
        trainer.dump_data_batch=false \
        trainer.eval_interval=100 \
        trainer.eval_before_train=false \
        trainer.eval_batch_size=32 \
        trainer.logger=console \
        trainer.project_name='skyrlsql' \
        trainer.run_name='$RUN_NAME' \
        trainer.ckpt_path='$CKPT_PATH' \
        trainer.export_path='$EXPORT_PATH' \
        trainer.log_path='$LOG_ROOT/$RUN_NAME' \
        ${EXTRA_OVERRIDES:-}
"
