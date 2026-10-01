# SkyRL latency parity with granular-cais-rl (SQL)

Workload: SkyRL-SQL (650 prompts), Qwen2.5-Coder-7B-Instruct, 4× L40S colocated, batch 256 × 5,
35 steps. Reference: granular-cais-rl `atc-exp18-record` (synchronous rollout→train, CPU Adam,
no streaming/compaction). SkyRL is matched to it: 24k token budget and max input, CUDA graphs,
vLLM GPU memory 0.9 / 2048 seqs, fp32 gradient reduce, seed 43, zero-signal groups dropped,
no KL / eval / checkpoints, 32 CPUs.

## Result (35 steps, mean s/step)

| | Stock SkyRL (job 6828066) | SkyRL + changes below (job 6905569) | granular exp18 |
|---|---:|---:|---:|
| **Step wall, end to end** | **537** ¹ | **443.0** | **422.8** |
| Rollout | 236.7 | 250.5 | 234.2 |
| Old-logprob pass | 49.3 | 0.0 | — |
| Training (incl. optimizer) | 215.5 | 187.1 | ~182 |
| Between steps (dataloader) | 30.4 | 0.1 | — |
| **vs granular** | **1.27×** | **1.05×** | 1.00× |

¹ includes ~30 s/step of dataloader respawn that SkyRL's `timing/step` does not count.

Training is at parity (within ~3%). The remaining ~16 s/step of rollout is within run-to-run
variance: granular's own exp17 and exp18 (identical config, different seed) differ by 13 s/step,
and none of the changes touch generation.

## Changes required

Code changes live on branch `dp-token-balance`; each is behind a flag.

| # | Change | Setting | Effect | Commit |
|---|---|---|---|---|
| 1 | CPU-resident fp32 Adam masters (granular's optimizer offload) | `trainer.policy.optimizer_config.cpu_adam=true` | Same memory model as granular; 24k-token microbatches fit beside vLLM | pre-`5b94dc06` |
| 2 | Zero-variance filter works with token-level rewards and truncates dropped trajectories (prompt and response) | `trainer.algorithm.zero_variance_filter=true` | Stops training on ~1.76× more tokens than granular | `d9efa278`, `95524f17` |
| 3 | Pinned host staging for the CPU-Adam grad copy and colocated model offload | `trainer.policy.fsdp_config.pinned_host_stage=true` (default) | Small (~8% of the optimizer step) | `7640da4a` |
| 4 | Token-balanced per-rank split of each mini-batch | `trainer.dp_token_balance=true` | Forward/backward per token = granular (0.481 → 0.429 s / 1k tokens); no padding microbatches | `40449520` |
| 5 | Skip the old-logprob recompute when there is one policy update per batch (ratio ≡ 1, same gradient) | `trainer.algorithm.on_policy_old_logprobs=true` | −49 s/step average | `c0be75ac` |
| 6 | `gc.freeze()` after policy-worker init | env `SKYRL_GC_FREEZE=1` | Removes a 20–180 s training-start stall (ranks starting the training call staggered) | `8998145f` |
| 7 | In-process prompt loading | `data.dataloader.num_workers=0` | Removes ~60 s worker respawn at every epoch boundary (every 2 steps here) | config only |

Launch script with everything: `/oscar/data/deeptir/mborjigi/skyrl/34_parity_final.sbatch`.

## Notes

- **Why each fix was needed:**
  - (4) Stock SkyRL gives each GPU a fixed quarter of the samples. FSDP's collectives run every
    microbatch in lockstep, so the whole step waits for the heaviest GPU, and lighter GPUs run
    padding microbatches.
  - (6) One thread per policy worker (Ray's async-actor event loop) sat at 100% CPU before the
    training call. Likely GC churn over ~545k long-lived objects. Upstream SkyRL applies the same
    `gc.freeze()` in its sharded weight-sync path (`freeze_trainer_heap`).
  - (7) The dataset has only 2 full batches per epoch, so the per-epoch respawn hits every other
    step.
- **Ruled out by a same-node benchmark:** fp32 vs bf16 weights on GPU, entropy computation
  (~2%), torch/transformers versions, and attention kernels. The model code itself runs at
  granular's speed.
- **Known residual:** the CPU Adam step is ~4–5 s/step slower than granular's (cause unconfirmed).
- **Before upstreaming:** drop the debug-only commits `28c536b7`, `ffb7490b`, `f5e6b322` (the
  `[stall-debug]` timing lines and the stack sampler). `8998145f` also carries the opt-in
  `SKYRL_GC_DEBUG` logging.
