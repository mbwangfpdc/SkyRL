"""Check fsdp_config.pinned_host_stage: pinned-staged copies must change timing, never results.

    torchrun --nproc_per_node=2 pinned_stage_check.py [--hidden 2048 --layers 16]

Runs the same randomly initialized Qwen2 through cpu_adam twice in one process -- once with
pinned_host_stage=False (the old pageable .to() paths), once with True -- each for STEPS steps of
MICRO_BATCHES micro-batches with clipping active, and after every step round-trips the model
through offload_to_cpu/backload_to_gpu the way colocate_all does between training and rollout.
Checks, per rank:

  * grad norms and final weights are BIT-EXACT between the two modes (staging only moves where
    the host copy lives; the same kernels do the same math);
  * weights are bit-identical before and after every offload/backload round trip;
  * the offload really frees the GPU weight shards (allocated memory drops by >= 90% of the local
    shard bytes) and backload restores them;
  * reports the optimizer-step and offload/backload timings of each mode.

Prints PINNED_STAGE_CHECK: PASS|FAIL.
"""

import argparse
import os
import time

import torch

STEPS = 3
MICRO_BATCHES = 3
SEQ = 64
VOCAB = 32000
LR = 1e-3
MAX_NORM = 0.05


def build_model(hidden: int, layers: int):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=16,
        num_key_value_heads=4,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16)


def local_weights(model):
    from torch.distributed.tensor import DTensor

    return {
        n: (p.to_local() if isinstance(p, DTensor) else p).detach().cpu().clone()
        for n, p in model.named_parameters()
    }


def run_mode(pinned: bool, hidden: int, layers: int, rank: int) -> dict:
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl.train.config import FSDPConfig, MixedPrecisionConfig, ModelConfig, OptimizerConfig

    fsdp_cfg = FSDPConfig(
        mixed_precision=MixedPrecisionConfig(param_dtype="bf16", reduce_dtype="fp32", buffer_dtype="bf16"),
        pinned_host_stage=pinned,
    )
    optim_cfg = OptimizerConfig(
        lr=LR, cpu_adam=True, master_dtype="fp32", max_grad_norm=MAX_NORM, offload_after_step=False
    )
    strategy = FSDPStrategy(fsdp_config=fsdp_cfg, optimizer_config=optim_cfg, model_config=ModelConfig())
    strategy.setup_distributed()
    model, optimizer, scheduler = strategy.prepare((build_model(hidden, layers), None, None))
    shard_bytes = sum(t.numel() * t.element_size() for t in local_weights(model).values())

    gen = torch.Generator().manual_seed(1234 + rank)
    res = {"norms": [], "t_opt": [], "t_off": [], "t_on": [], "roundtrip_ok": True, "freed_frac": []}
    for _ in range(STEPS):
        for _ in range(MICRO_BATCHES):
            ids = torch.randint(0, VOCAB, (2, SEQ), generator=gen).cuda()
            loss = model(input_ids=ids, labels=ids).loss / MICRO_BATCHES
            strategy.backward(loss, model, optimizer)
        torch.cuda.synchronize()
        t0 = time.time()
        res["norms"].append(float(strategy.optimizer_step(optimizer, model, scheduler)))
        torch.cuda.synchronize()
        res["t_opt"].append(time.time() - t0)

        before = local_weights(model)
        alloc0 = torch.cuda.memory_allocated()
        t0 = time.time()
        strategy.offload_to_cpu(model, optimizer)
        res["t_off"].append(time.time() - t0)
        res["freed_frac"].append((alloc0 - torch.cuda.memory_allocated()) / shard_bytes)
        t0 = time.time()
        strategy.backload_to_gpu(model, optimizer)
        res["t_on"].append(time.time() - t0)
        after = local_weights(model)
        res["roundtrip_ok"] &= all(torch.equal(before[n], after[n]) for n in before)
    res["weights"] = local_weights(model)
    res["shard_gib"] = shard_bytes / 2**30
    del model, optimizer, scheduler, strategy
    torch.cuda.empty_cache()
    return res


def main():
    import torch.distributed as dist

    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=16)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()

    base = run_mode(False, args.hidden, args.layers, rank)
    pin = run_mode(True, args.hidden, args.layers, rank)

    checks = {
        "grad norms bit-exact": base["norms"] == pin["norms"],
        "final weights bit-exact": all(torch.equal(base["weights"][n], pin["weights"][n]) for n in base["weights"]),
        "pageable offload round trip exact": base["roundtrip_ok"],
        "pinned offload round trip exact": pin["roundtrip_ok"],
        "pinned offload frees >=90% of shard": min(pin["freed_frac"]) >= 0.9,
    }
    ok = all(checks.values())
    flags = [None] * dist.get_world_size()
    dist.all_gather_object(flags, ok)

    def fmt(xs):
        return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"

    for r in range(dist.get_world_size()):
        if r == rank:
            print(f"--- rank {rank}: local weight shard {pin['shard_gib']:.2f} GiB")
            for name, v in checks.items():
                print(f"  {'ok  ' if v else 'FAIL'} {name}")
            print(f"  norms pageable={base['norms']} pinned={pin['norms']}")
            for label, key in (("optimizer_step", "t_opt"), ("offload", "t_off"), ("backload", "t_on")):
                print(f"  {label:<15} s: pageable={fmt(base[key])} pinned={fmt(pin[key])}")
            print(f"  freed fraction of shard on offload: pageable={fmt(base['freed_frac'])} pinned={fmt(pin['freed_frac'])}")
        dist.barrier()
    if rank == 0:
        print(f"PINNED_STAGE_CHECK: {'PASS' if all(flags) else 'FAIL'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
