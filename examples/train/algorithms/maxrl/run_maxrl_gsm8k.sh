set -x

# Colocated MaxRL training+generation for Qwen2.5-7B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
export WANDB_API_KEY=7f550638bde3906c1462f6ee02aab7b5cb000ebb
# bash examples/gsm8k/run_gsm8k.sh


# You can override the default values with e.g.: `NUM_GPUS=1 bash examples/train/algorithms/maxrl/run_maxrl_gsm8k.sh`.

: "${DATA_DIR:="/local_ssd1/mborjigi/skyrl/SkyRL/skyrl-train/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=console}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"

# MAXRL parameters
: "${ADV_ESTIMATOR:=maxrl}"

# Other algorithm parameters
: "${USE_KL_LOSS:=true}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="$ADV_ESTIMATOR" \
  trainer.policy.model.path="Qwen/Qwen2.5-7B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.colocate_policy_ref=true \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=32 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="maxrl_gsm8k" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="/local_ssd1/mborjigi/tmp/ckpts/gsm8k_7B_ckpt" \
  $@
