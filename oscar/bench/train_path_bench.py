"""Per-microbatch forward/backward timing: SkyRL's training path vs granular-cais-rl's, same inputs.

    torchrun --standalone --nproc_per_node=4 train_path_bench.py --mode {skyrl,granular} [flags]

Every mode trains Qwen2.5-Coder-7B-Instruct under FSDP2 (per-decoder-layer + root, param bf16,
reduce fp32, non-reentrant activation checkpointing, flash-attn varlen on packed rows) on the SAME
seeded packed microbatches (~TOKENS tokens each, sequence lengths 1200-4000), WARMUP untimed then
MEASURE timed, each with a synchronize before/after forward and backward.

  --mode skyrl      SkyRL's own HFModelWrapper + FSDPStrategy.prepare exactly as its policy worker
                    builds them (fp32 load unless --bf16-load; compute_entropy=True unless
                    --no-entropy; padded [B, S] batch unpadded inside the wrapper; logits.div_(temp);
                    flash-attn cross-entropy logprobs), PPO-ratio loss against detached logprobs.
  --mode granular   granular's path: bf16 load, trunk forward with lm_head swapped to Identity and
                    precomputed cu_seq_lens/max_length, manual lm_head, chunked selective-logsumexp
                    logprob (copied from train_worker._SelectiveLogprob), -A*logp token-sum loss.
                    Imports only torch/transformers, so it also runs in granular's own venv.
"""
import argparse
import itertools
import os
import statistics as stt
import time

import torch
import torch.distributed as dist

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEMP = 0.6


def make_batches(n, tokens, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        lens = []
        while True:
            L = int(torch.randint(1200, 4001, (1,), generator=g))
            if sum(lens) + L > tokens:
                break
            lens.append(L)
        ids = [torch.randint(100, 150000, (L,), generator=g) for L in lens]
        out.append((lens, ids))
    return out


# --- granular's selective logprob (train_worker._SelectiveLogprob, verbatim math) ---------------
_LSE_SLOTS = 4096


class _SelectiveLogprob(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, temp=1.0):
        chunk = max(1, _LSE_SLOTS // max(1, logits.shape[0]))
        lse = torch.cat(
            [torch.logsumexp(c.float().div_(temp), dim=-1) for c in logits.split(chunk, dim=1)], dim=1
        )
        tok_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1).float().div_(temp)
        ctx.save_for_backward(logits, lse, target)
        ctx.temp = temp
        return tok_logit - lse

    @staticmethod
    def backward(ctx, grad_out):
        logits, lse, target = ctx.saved_tensors
        temp = ctx.temp
        chunk = max(1, _LSE_SLOTS // max(1, logits.shape[0]))
        for s in range(0, logits.shape[1], chunk):
            e = min(s + chunk, logits.shape[1])
            c = logits[:, s:e]
            g = (grad_out[:, s:e] / temp).unsqueeze(-1)
            sm = (c.float().div_(temp) - lse[:, s:e].unsqueeze(-1)).exp_()
            sm.mul_(-g)
            sm.scatter_add_(-1, target[:, s:e].unsqueeze(-1), g.to(sm.dtype))
            c.copy_(sm.to(logits.dtype))
        return logits, None, None


def build_granular(dev, world):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = model.to(dev)
    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                              output_dtype=torch.bfloat16, cast_forward_inputs=True)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    model.train()

    def step(lens, ids):
        in_ids = torch.cat(ids).unsqueeze(0).to(dev)
        pos = torch.cat([torch.arange(L) for L in lens]).unsqueeze(0).to(dev)
        cu = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device=dev)
        adv = torch.randn(in_ids.shape, device=dev)[:, 1:]
        mask = torch.ones_like(adv)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lm_head = model.lm_head
            model.lm_head = torch.nn.Identity()
            try:
                out = model(input_ids=in_ids, attention_mask=None, position_ids=pos,
                            cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max(lens), max_length_k=max(lens))
            finally:
                model.lm_head = lm_head
            hidden = out.logits[:, :-1]
            logits = lm_head(hidden.to(torch.bfloat16))
            tok_lp = _SelectiveLogprob.apply(logits, in_ids[:, 1:], TEMP)
            loss = -(adv * tok_lp * mask).sum() * (world / 24000)
        torch.cuda.synchronize(); t1 = time.time()
        loss.backward()
        torch.cuda.synchronize(); t2 = time.time()
        return t1 - t0, t2 - t1

    return model, step


def build_skyrl(dev, world, bf16_load, entropy):
    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.train.config import FSDPConfig, MixedPrecisionConfig, ModelConfig, OptimizerConfig

    fsdp_cfg = FSDPConfig(mixed_precision=MixedPrecisionConfig(param_dtype="bf16", reduce_dtype="fp32"))
    optim_cfg = OptimizerConfig(lr=1e-6, cpu_adam=True, max_grad_norm=0.5, offload_after_step=False)
    strategy = FSDPStrategy(fsdp_config=fsdp_cfg, optimizer_config=optim_cfg, model_config=ModelConfig())
    strategy.setup_distributed()
    wrapped = HFModelWrapper(MODEL, use_flash_attention_2=True, bf16=bf16_load,
                             remove_microbatch_padding=True, logprobs_chunk_size=1024)
    wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model, _, _ = strategy.prepare((wrapped, None, None))
    # As the policy worker does right after prepare (Worker._set_expandable_segments(True)).
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    model.train()

    def step(lens, ids):
        B, S = len(lens), max(lens)
        seq = torch.zeros(B, S, dtype=torch.long)
        am = torch.zeros(B, S, dtype=torch.long)
        for i, t in enumerate(ids):  # left-pad, as SkyRL batches do
            seq[i, S - len(t):] = t
            am[i, S - len(t):] = 1
        seq, am = seq.to(dev), am.to(dev)
        num_actions = S - 1
        adv = torch.randn(B, num_actions, device=dev)
        mask = am[:, 1:].float()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            lp, _ = model(seq, num_actions, attention_mask=am, temperature=TEMP, return_output=True,
                          compute_entropy=entropy, entropy_requires_grad=False)
            ratio = torch.exp(lp - lp.detach())
            loss = -(adv * ratio * mask).sum() / mask.sum()
        torch.cuda.synchronize(); t1 = time.time()
        strategy.backward(loss, model, None)
        torch.cuda.synchronize(); t2 = time.time()
        return t1 - t0, t2 - t1

    return model, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["skyrl", "granular"], required=True)
    ap.add_argument("--bf16-load", action="store_true")
    ap.add_argument("--no-entropy", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokens", type=int, default=22000)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measure", type=int, default=10)
    args = ap.parse_args()

    dist.init_process_group("nccl")  # FSDPStrategy.setup_distributed reads the existing group
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)

    if args.mode == "granular":
        model, step = build_granular(dev, world)
    else:
        model, step = build_skyrl(dev, world, args.bf16_load, not args.no_entropy)

    batches = make_batches(args.warmup + args.measure, args.tokens, seed=1234 + rank)
    fw, bw, tok = [], [], []
    for i, (lens, ids) in enumerate(batches):
        f, b = step(lens, ids)
        model.zero_grad(set_to_none=True)
        if i >= args.warmup:
            fw.append(f); bw.append(b); tok.append(sum(lens))
    peak = torch.cuda.max_memory_allocated() / 2**30
    res = torch.tensor([stt.mean(fw), stt.mean(bw), stt.mean(tok), peak], device=dev)
    allr = [torch.zeros_like(res) for _ in range(world)]
    dist.all_gather(allr, res)
    if rank == 0:
        import torch as _t, transformers as _tf
        a = torch.stack(allr).cpu()
        f, b, t, pk = a[:, 0].max().item(), a[:, 1].max().item(), a[:, 2].mean().item(), a[:, 3].max().item()
        print(f"BENCH mode={args.mode}{args.tag} torch={_t.__version__} transformers={_tf.__version__} "
              f"tokens/mb={t:.0f} fwd={f:.3f}s bwd={b:.3f}s fb={f + b:.3f}s "
              f"per1k fwd={f / t * 1e3:.4f}s bwd={b / t * 1e3:.4f}s peak={pk:.1f}GiB", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
