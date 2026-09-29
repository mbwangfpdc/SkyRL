"""Equivalence + memory check for optimizer_config.stream_grads_to_cpu (ZeRO-Offload-style).

Trains the same randomly-initialized tiny Qwen2 twice through FSDPStrategy -- cpu_adam with the
post-backward bulk grad copy (reference) vs. cpu_adam + per-layer gradient streaming -- over a few
optimizer steps with gradient accumulation and clipping forced active, then compares the reported
grad norms, the step-1 CPU gradients element-wise (the strict check: both runs start from
identical weights there), the final weights (loose: Adam's m/sqrt(v) amplifies rounding-level
differences in near-zero grads to +-lr, so later steps drift apart legitimately), and peak GPU
memory across forward+backward (the model is sized so gradients, not activations, dominate).

Not a pytest file (needs >=2 GPUs under torchrun):

    torchrun --nproc_per_node=2 tests/backends/skyrl_train/gpu/stream_grads_equivalence_check.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import Qwen2Config, Qwen2ForCausalLM

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.train.config import FSDPConfig, ModelConfig, OptimizerConfig

STEPS = 3
MICRO_BATCHES = 3
SEQ = 64
VOCAB = 32000


def build_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).to(torch.float32)


def run(stream: bool):
    optim_cfg = OptimizerConfig(
        lr=1e-3, cpu_adam=True, stream_grads_to_cpu=stream, max_grad_norm=0.05, offload_after_step=False
    )
    strategy = FSDPStrategy(fsdp_config=FSDPConfig(), optimizer_config=optim_cfg, model_config=ModelConfig())
    strategy.setup_distributed()
    model, optimizer, scheduler = strategy.prepare((build_model(), None, None))

    # Snapshot the CPU masters' grads (post-clip, what AdamW actually consumes) at each step.
    grad_snaps = []
    orig_step = optimizer.step

    def step_with_snapshot(*args, **kwargs):
        grad_snaps.append(
            {n: m.grad.detach().clone() for n, m in strategy._cpu_master.items() if m.grad is not None}
        )
        return orig_step(*args, **kwargs)

    optimizer.step = step_with_snapshot

    rank = dist.get_rank()
    gen = torch.Generator().manual_seed(1234 + rank)
    norms, peaks, post_bwd = [], [], []
    for _ in range(STEPS):
        torch.cuda.reset_peak_memory_stats()
        for _ in range(MICRO_BATCHES):
            ids = torch.randint(0, VOCAB, (2, SEQ), generator=gen).cuda()
            loss = model(input_ids=ids, labels=ids).loss / MICRO_BATCHES
            strategy.backward(loss, model, optimizer)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() / 2**20)
        # Allocated right after backward: includes the full sharded grads in the reference path.
        post_bwd.append(torch.cuda.memory_allocated() / 2**20)
        norms.append(float(strategy.optimizer_step(optimizer, model, scheduler)))
    weights = {
        n: (p.to_local() if isinstance(p, DTensor) else p).detach().float().cpu().clone()
        for n, p in model.named_parameters()
    }
    del model, optimizer, scheduler, strategy
    torch.cuda.empty_cache()
    return norms, weights, peaks, grad_snaps, post_bwd


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()

    # Two reference runs: bf16 forward/backward kernels are not bit-deterministic, so ref-vs-ref
    # gives the noise floor that ref-vs-stream is judged against.
    ref_norms, ref_w, ref_peak, ref_g, ref_post = run(stream=False)
    ref2_norms, _, _, ref2_g, _ = run(stream=False)
    st_norms, st_w, st_peak, st_g, st_post = run(stream=True)

    def grad_rel(a, b):
        if set(a) != set(b):
            return float("inf")
        scale = max(g.abs().max().item() for g in a.values())
        return max((a[n] - b[n]).abs().max().item() for n in a) / max(scale, 1e-30)

    same_keys = set(ref_g[0]) == set(st_g[0])
    g0_rel = grad_rel(ref_g[0], st_g[0])
    g0_floor = grad_rel(ref_g[0], ref2_g[0])
    norm0_rel = abs(ref_norms[0] - st_norms[0]) / ref_norms[0]
    norm0_floor = abs(ref_norms[0] - ref2_norms[0]) / ref_norms[0]

    max_w_diff = max((ref_w[n] - st_w[n]).abs().max().item() for n in ref_w)
    max_w_scale = max(ref_w[n].abs().max().item() for n in ref_w)
    norm_rel = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(ref_norms, st_norms))
    if rank == 0:
        print(f"grad_norm ref   : {ref_norms}")
        print(f"grad_norm stream: {st_norms}")
        print(f"grad_norm max rel diff: {norm_rel:.3e}")
        print(f"grad_norm ref2  : {ref2_norms}")
        print(f"step-1 grads ({len(ref_g[0])} tensors, keys match={same_keys}): "
              f"ref-vs-stream rel {g0_rel:.3e}, ref-vs-ref floor {g0_floor:.3e}")
        print(f"step-1 norm: ref-vs-stream rel {norm0_rel:.3e}, ref-vs-ref floor {norm0_floor:.3e}")
        print(f"weights max abs diff: {max_w_diff:.3e} (max |w| {max_w_scale:.3e})")
        print(f"peak GPU MiB ref   : {[round(x) for x in ref_peak]}")
        print(f"peak GPU MiB stream: {[round(x) for x in st_peak]}")
        print(f"post-backward allocated MiB ref   : {[round(x) for x in ref_post]}")
        print(f"post-backward allocated MiB stream: {[round(x) for x in st_post]}")
    # Strict: step-1 grads element-wise and step-1 norm (identical starting weights) within a
    # small multiple of the ref-vs-ref kernel-nondeterminism floor (plus fp32 epsilon slack).
    # Loose: later norms, and weights within one Adam step (lr * STEPS) -- see module docstring.
    # all(n > max_grad_norm) confirms clipping was actually exercised on every step.
    # Memory: after backward the stream path must hold less than the reference (no GPU grads).
    ok = (
        same_keys
        and g0_rel <= 3 * g0_floor + 1e-6
        and norm0_rel <= 3 * norm0_floor + 1e-6
        and st_post[-1] < ref_post[-1]
        and norm_rel < 1e-2
        and max_w_diff <= 1e-3 * STEPS * 1.01
        and all(n > 0.05 for n in ref_norms)
    )
    ok_t = torch.tensor(int(ok), device="cuda")
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("STREAM_GRADS_EQUIVALENCE:", "PASS" if ok_t.item() else "FAIL")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
