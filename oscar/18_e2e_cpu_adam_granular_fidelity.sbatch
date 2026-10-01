#!/bin/bash
# Real e2e SkyRL-SQL training run using cpu_adam ("optimizer offload"), tuned
# for closest fidelity to a typical granular-cais-rl run -- NOT a matched-
# comparison-fairness run (see granular-skyrl-comparison-fairness memory,
# which is a different, narrower project with its own opposite requirements
# on reduce_dtype and dynamic_sampling).
#
# Branch: cpu-adam-on-repro (torch==2.11.0/vllm==0.23.0/ray==2.56.0, matches
# granular; has both cpu_adam and max_tokens_per_microbatch -- see
# skyrl-cpu-adam-port memory for why track-mbwang was the wrong branch for
# this). SRC intentionally NOT overridden -- the primary checkout is already
# on this branch.
#
# REVISED 2026-09-10 (superseding the first e2e attempt, job 6191717,
# COMPLETED but using the wrong mechanism): drop zero-signal groups now uses
# trainer.algorithm.zero_variance_filter=true instead of
# dynamic_sampling.type=filter. Investigation found these are NOT equivalent:
# dynamic_sampling.type="filter" is DAPO's own mechanism -- it actively
# RESAMPLES (draws fresh prompts and runs new rollouts) to backfill dropped
# groups until the batch is full, which is what SkyRL's own docs describe as
# "dynamic sampling." Reading granular's actual streaming.py/main.py showed
# its drop_zero_signal is a much simpler, passive mechanism: one fixed batch
# draw per step, zero-variance groups just excluded from training with NO
# backfill/resampling -- the effective batch size silently shrinks instead.
# zero_variance_filter is the real analog of that. Confirmed by reading
# ppo_utils.py's normalization math: the default loss_reduction="token_mean"
# divides by loss_mask.sum() (only unmasked tokens), computed once per
# mini-batch (== the whole round here, since policy_mini_batch_size ==
# train_batch_size), so masked-out groups are correctly excluded from both
# the loss numerator (already true anyway -- GRPO advantage is exactly 0 for
# a zero-variance group) and the mean-normalization denominator, matching
# granular's own "rescale to the exact global token mean" behavior.
#
# zero_variance_filter also needed two real fixes (both applied and pushed
# to cpu-adam-on-repro, commit d9efa278, before this run) to actually behave
# like granular's mechanism rather than just gradient-correctly:
#   1. It only ever loss-masked the full original completion length, so
#      dropped trajectories still paid full forward/backward compute for a
#      zero-gradient contribution -- not what "reduce compute" means.
#      Trainer.postprocess_generator_output now truncates dropped
#      trajectories to a 1-token stub instead (trajectory *count* stays
#      fixed -- compute_prompt_mini_batch_boundaries requires exactly
#      train_batch_size prompts -- only content shrinks). Verified via a
#      real GPU smoke test (job 6200622): 3 clean steps, checkpoints saved
#      fine, no shape-mismatch crashes.
#   2. It was silently a no-op for SkyRL-SQL specifically: the multi-turn
#      agent loop (any environment without custom_chat_template) returns
#      token-level rewards, and zero_variance_filter's caller only ever
#      handled response-level (scalar) rewards. First smoke-test attempt
#      (job 6197781) completed 3 steps with the filter metric never once
#      appearing -- proof it never fired. Fixed by collapsing token-level
#      rewards to a per-trajectory scalar (sum) before the std/group check,
#      the same pattern handle_filter_sampling/handle_replace_sampling
#      already use for the same purpose.
#
# Settings confirmed against the user's 5 requirements, 2026-09-10:
#   - drop zero-signal groups -> trainer.algorithm.zero_variance_filter=true,
#     trainer.algorithm.zero_variance_filter_tol=0.0 (exact std==0 equality,
#     matching granular's own `std_val > 1e-8` check -- our SQL/DAPO rewards
#     are exact 0/1/-1 values, not noisy float LLM-judge scores, so the
#     default 1e-6 tolerance would have been fine too, but 0.0 is the more
#     literal match). This is granular's own drop_zero_signal=true default;
#     NOT the same as the matched-comparison project's drop_zero_signal=false.
#   - reduce_dtype=bf16 -> trainer.policy.fsdp_config.mixed_precision.reduce_dtype=bf16.
#     Left UNSET, every prior script in this investigation silently got fp32
#     (the code's own hardcoded fallback when mixed_precision is never
#     configured -- see _fsdp_init_model). The fp32 requirement in
#     granular-skyrl-comparison-fairness was specific to that matched-
#     comparison project (eliminating an unrelated communication-cost
#     advantage); granular's own standalone default is bf16, cheaper, and
#     legitimate on its own -- this is what "closest fidelity to typical
#     granular-cais-rl runs" calls for. Only set on policy: ref never runs
#     backward, so its reduce_dtype never matters.
#   - token budget -> max_tokens_per_microbatch=28000. Already validated at
#     this exact topology (4 GPUs, cpu_adam=true, colocate_all=true, this
#     7B model) by job 5743785 -- not a new guess.
#   - optimizer offload enabled -> optimizer_config.cpu_adam=true (paired
#     with fsdp_config.cpu_offload=false, since the two are mutually
#     exclusive -- see the cpu_adam port's own ValueError guard). This is
#     specifically cpu_adam, not offload_after_step or native FSDP2
#     cpu_offload.
#   - 18 epochs -> trainer.epochs=18. Matches every script this session has
#     used already.
#
# Additional changes needed because this is a REAL run, not a smoke test:
#   - trainer.max_training_steps=35, back on (was set to 35, then removed
#     entirely on job 6191717 -- now restored with a corrected understanding
#     of why it failed before). Under dynamic_sampling.type="filter", each
#     real step could consume several resample rounds, so max_training_steps
#     capped raw batch draws rather than real steps (job 6189556 hit its
#     35-batch cap after only 19 real steps and exited early). With
#     dynamic_sampling now unset (None) -- zero_variance_filter needs no
#     resampling, it's a single fixed batch draw per step -- that ambiguity
#     is gone: handle_dynamic_sampling always returns keep_sampling=False
#     immediately when type is None, so every raw batch draw IS exactly one
#     real step, deterministically. Dataset is 650 rows, train_batch_size=256,
#     dataloader uses drop_last=True -> 2 raw batches/epoch x 18 epochs = 36
#     real steps available if uncapped, comfortably more than the 35 target
#     (1 step of margin) -- max_training_steps=35 will now correctly stop the
#     run at exactly step 35, matching granular's own 35-step reference
#     recipe (sql_baseline.toml/sql_optimized.toml: max_steps=35,
#     batch_size=256, num_generations=5, drop_zero_signal=true -- all of
#     which this run's train_batch_size=256/n_samples_per_prompt=5/
#     zero_variance_filter=true already match per-step).
#   - New run_name (skyrlsql_cpu_adam_e2e_zerovar, was skyrlsql_cpu_adam_e2e)
#     with trainer.resume_mode=latest kept on: the earlier e2e run (job
#     6191717, COMPLETED at step 22) trained under the OLD, wrong mechanism
#     (dynamic_sampling.type="filter", DAPO-style active resampling) --
#     resuming that checkpoint into this differently-configured run would
#     silently mix training regimes mid-run. A new run_name gives a fresh,
#     empty ckpt_path, so resume_mode=latest correctly falls back to
#     "starting training from scratch" on first submission (confirmed by
#     reading trainer.py's checkpoint-path resolution: it only looks for
#     `latest_ckpt_global_step.txt` under ckpt_path and no-ops if absent) --
#     while still auto-resuming from THIS run's own checkpoints if it needs
#     resubmitting to finish within one job's wall clock.
#   - --time=24:00:00 + trainer.ckpt_interval=10: one job submission's worth
#     of wall clock, checkpointing often enough that a requeue loses little
#     progress. cpu_adam's optimizer state (the CPU master Adam moments)
#     round-trips through the existing per-rank optimizer.state_dict()
#     checkpoint save/load unchanged -- verified during the port. Resubmit
#     this same script (resume_mode=latest picks up automatically) if 18
#     epochs needs more than one job's wall clock -- check the actual
#     gpu-he+ per-job wall-clock cap if 24h turns out to be constrained.
#   - trainer.eval_interval=10 (was disabled/1000 in smoke tests), matching
#     granular's own eval_every=10 default, for the same fidelity reason as
#     reduce_dtype above.
#   - STAGE_DB=0 kept (not switched to 1): this is the already-proven-working
#     default from this session (see 14_matched_2step_accounting.sbatch's
#     header for the two DB-staging hangs it replaced). A long run might
#     benefit from local staging's faster repeated small-file access, but
#     that's untested here -- flagged, not assumed. Override with
#     STAGE_DB=1 if NFS read latency turns out to matter over 18 epochs.

#SBATCH --job-name=skyrl-cpu-adam-e2e-zerovar
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768g
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu2709,gpu3006
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail
source /oscar/scratch/mborjigi/skyrl/env.sh
skyrl_set_jobtmp

SRC="$SKYRL_ROOT/SkyRL"

export SKYRL_GPU=1

NUM_GPUS=4
NUM_INFERENCE_ENGINES=4
TP_SIZE=1
MAX_INPUT_LENGTH=8192
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256

RUN_NAME=${RUN_NAME:-skyrlsql_cpu_adam_e2e_zerovar}
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
if ! grep -q "cpu_adam" "$SRC/skyrl/train/config/config.py"; then
    echo "FATAL: cpu_adam not found in $SRC/skyrl/train/config/config.py -- checkout is not what it claims to be."
    exit 1
fi
if ! grep -q "_apply_zero_variance_filter" "$SRC/skyrl/train/trainer.py"; then
    echo "FATAL: the zero_variance_filter compute-truncation patch is not present in $SRC/skyrl/train/trainer.py -- refusing to run without it."
    exit 1
fi
if ! grep -q "seq_rewards = \[float(sum(r)) for r in rewards\]" "$SRC/skyrl/train/trainer.py"; then
    echo "FATAL: the token-level-reward fix for zero_variance_filter is not present in $SRC/skyrl/train/trainer.py -- refusing to run without it (SkyRL-SQL's multi-turn agent loop needs this, or the filter silently never fires)."
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

echo "=== scratch usage before start ==="
checkquota 2>/dev/null | awk '/oscar.scratch/ {print}' || df -h /oscar/scratch/mborjigi

echo "=== starting ray head + SkyRL-SQL (cpu_adam=true, zero_variance_filter=true, e2e, granular fidelity) ==="
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
        trainer.max_training_steps=35 \
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
        trainer.ckpt_interval=10 \
        trainer.max_ckpts_to_keep=3 \
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
        trainer.resume_mode=latest \
        trainer.ckpt_path='$CKPT_PATH' \
        trainer.export_path='$EXPORT_PATH' \
        trainer.log_path='$LOG_ROOT/$RUN_NAME' \
        trainer.eval_batch_size=32 \
        trainer.eval_before_train=false \
        trainer.eval_interval=10 \
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

echo "job finished (exit=$TRAIN_EXIT). Re-submit this script to resume from $CKPT_PATH"
exit "$TRAIN_EXIT"
