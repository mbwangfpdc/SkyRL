"""Functionality check: cpu_adam bulk-copy vs cpu_adam + stream_grads_to_cpu (ZeRO-Offload-style).

Two separate tests, one process each, same randomly-initialized ~200M-param Qwen2 and the same
data, everything bf16 except Adam (weights, compute, reduce-scatter and gradients in bf16; CPU
AdamW masters/moments in fp32), 3 steps x 3 micro-batches with clipping forced active:

    torchrun --nproc_per_node=2 stream_grads_equivalence_check.py run --mode cpu_adam --out DIR
    torchrun --nproc_per_node=2 stream_grads_equivalence_check.py run --mode stream   --out DIR
    python stream_grads_equivalence_check.py compare --out DIR

`run` saves, per rank: reported (pre-clip) grad norms, the step-1 accumulated pre-clip gradients
(GPU .grad for cpu_adam, the drained CPU accumulators for stream), the step-1 CPU gradients AdamW
consumed,
final weights, per-step timings, and GPU memory (peak and right after backward, relative to the
run's starting allocation). `compare` checks:

  * step-1 accumulated pre-clip gradients: identical starting weights and data, and streaming's
    accumulate-via-GPU does the same bf16 adds as FSDP2's on-GPU accumulation, so these must be
    BIT-EXACT.
  * step-1 clipped gradients (what AdamW consumed): agree to bf16 rounding -- the bulk-copy path
    clips in bf16 with a bf16-rounded norm, streaming with an fp32 norm fused into the fp32
    upcast. Compared after undoing each run's clip (* (norm + 1e-6) / max_norm).
  * step-1 norm within bf16 precision; later steps drift legitimately (Adam's m/sqrt(v) amplifies
    rounding-level grad differences to +-lr per element per step, in either direction), so later
    norms and final weights (bound 2 * lr * steps) are loose checks.

`--tag` names a run's result files (default: the mode); `compare --floor TAG` also loads a second
cpu_adam run saved under TAG and judges the pre-clip and clipped gradient checks against the
cpu_adam-vs-cpu_adam difference instead of bit-exactness (at longer sequences the attention and
embedding backward kernels are not bit-deterministic run to run).

`--seq` sets tokens per micro-batch (default 64: correctness; e.g. 4096: timing with realistic
backward compute to overlap the transfers with).
  * memory: after backward the streaming run must hold less GPU memory (no resident grads).
"""

import argparse
import gc
import os
import time

import torch

STEPS = 3
MICRO_BATCHES = 3
SEQ = 64
VOCAB = 32000
LR = 1e-3
MAX_NORM = 0.05


def build_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

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
    # Init in fp32 then cast, so both modes get bit-identical bf16 starting weights.
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16)


def run(mode: str, out: str, seq: int, tag: str):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
    from skyrl.train.config import FSDPConfig, MixedPrecisionConfig, ModelConfig, OptimizerConfig

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    gc.collect()
    torch.cuda.empty_cache()
    base_mib = torch.cuda.memory_allocated() / 2**20

    fsdp_cfg = FSDPConfig(
        mixed_precision=MixedPrecisionConfig(param_dtype="bf16", reduce_dtype="bf16", buffer_dtype="bf16")
    )
    optim_cfg = OptimizerConfig(
        lr=LR,
        cpu_adam=True,
        master_dtype="fp32",
        stream_grads_to_cpu=(mode == "stream"),
        max_grad_norm=MAX_NORM,
        offload_after_step=False,
    )
    strategy = FSDPStrategy(fsdp_config=fsdp_cfg, optimizer_config=optim_cfg, model_config=ModelConfig())
    strategy.setup_distributed()
    model, optimizer, scheduler = strategy.prepare((build_model(), None, None))

    dtypes = {str(p.dtype) for p in model.parameters()}
    assert dtypes == {"torch.bfloat16"}, f"model params not all bf16: {dtypes}"

    # Snapshot the CPU masters' grads (post-clip, what AdamW actually consumes) at each step.
    grad_snaps = []
    orig_step = optimizer.step

    def step_with_snapshot(*args, **kwargs):
        grad_snaps.append(
            {n: m.grad.detach().clone() for n, m in strategy._cpu_master.items() if m.grad is not None}
        )
        return orig_step(*args, **kwargs)

    optimizer.step = step_with_snapshot

    gen = torch.Generator().manual_seed(1234 + rank)
    norms, peaks, post_bwd, t_fb, t_opt, grad_dtypes = [], [], [], [], [], set()
    pre_clip0 = None
    for step in range(STEPS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(MICRO_BATCHES):
            ids = torch.randint(0, VOCAB, (2, seq), generator=gen).cuda()
            loss = model(input_ids=ids, labels=ids).loss / MICRO_BATCHES
            strategy.backward(loss, model, optimizer)
        torch.cuda.synchronize()
        t1 = time.time()
        peaks.append(torch.cuda.max_memory_allocated() / 2**20 - base_mib)
        post_bwd.append(torch.cuda.memory_allocated() / 2**20 - base_mib)
        if mode == "stream":
            grad_dtypes |= {str(g.dtype) for g in strategy._cpu_grad.values()}
        else:
            grad_dtypes |= {str(p.grad.dtype) for p in model.parameters() if p.grad is not None}
        if step == 0:  # outside both timed windows
            if mode == "stream":
                torch.cuda.synchronize()
                pre_clip0 = {n: strategy._cpu_grad[n].clone() for n in strategy._cpu_grad_filled}
            else:
                pre_clip0 = {
                    n: (p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).detach().cpu().clone()
                    for n, p in model.named_parameters()
                    if p.grad is not None
                }
            torch.cuda.synchronize()
        t2 = time.time()
        norms.append(float(strategy.optimizer_step(optimizer, model, scheduler)))
        torch.cuda.synchronize()
        t_fb.append(t1 - t0)
        t_opt.append(time.time() - t2)

    weights = {
        n: (p.to_local() if isinstance(p, DTensor) else p).detach().cpu().clone()
        for n, p in model.named_parameters()
    }
    os.makedirs(out, exist_ok=True)
    torch.save(
        {
            "mode": mode,
            "norms": norms,
            "grads0": grad_snaps[0],
            "pre_clip0": pre_clip0,
            "seq": seq,
            "weights": weights,
            "peak_mib": peaks,
            "post_bwd_mib": post_bwd,
            "t_fwd_bwd": t_fb,
            "t_opt": t_opt,
            "grad_dtypes": sorted(grad_dtypes),
        },
        os.path.join(out, f"{tag}_rank{rank}.pt"),
    )
    if rank == 0:
        print(f"[{mode}] seq={seq}")
        print(f"[{mode}] norms={norms} grad_dtypes={sorted(grad_dtypes)}")
        print(f"[{mode}] peak MiB={[round(x) for x in peaks]} post-backward MiB={[round(x) for x in post_bwd]}")
        print(f"[{mode}] fwd+bwd s={[round(x, 3) for x in t_fb]} optimizer_step s={[round(x, 3) for x in t_opt]}")
    dist.destroy_process_group()


def _pre_clip_diff(pa, pb):
    """(#tensors not bit-exact, max abs diff, max per-tensor relative L2 diff) between two runs."""
    if set(pa) != set(pb):
        return len(pa), float("inf"), float("inf")
    n_bad = sum(not torch.equal(pa[n], pb[n]) for n in pa)
    max_abs = max(((pa[n].float() - pb[n].float()).abs().max().item() for n in pa), default=0.0)
    max_rel = max(
        (((pa[n].float() - pb[n].float()).norm() / pa[n].float().norm().clamp_min(1e-30)).item() for n in pa),
        default=0.0,
    )
    return n_bad, max_abs, max_rel


def compare(out: str, floor: str = None) -> bool:
    ranks = sorted(int(f.split("rank")[1][:-3]) for f in os.listdir(out) if f.startswith("cpu_adam_rank"))
    assert ranks, f"no results in {out}"
    ok = True
    for r in ranks:
        a = torch.load(os.path.join(out, f"cpu_adam_rank{r}.pt"))
        b = torch.load(os.path.join(out, f"stream_rank{r}.pt"))
        assert a["seq"] == b["seq"], (a["seq"], b["seq"])
        pa, pb = a["pre_clip0"], b["pre_clip0"]
        pre_keys_ok = set(pa) == set(pb)
        pre_bad, pre_max, pre_rel = _pre_clip_diff(pa, pb)
        fl = None
        if floor:
            f = torch.load(os.path.join(out, f"{floor}_rank{r}.pt"))
            assert f["seq"] == a["seq"], (f["seq"], a["seq"])
            fl_bad, fl_max, fl_rel = _pre_clip_diff(pa, f["pre_clip0"])
            fg = f["grads0"]
            uf = (f["norms"][0] + 1e-6) / MAX_NORM
            ua_ = (a["norms"][0] + 1e-6) / MAX_NORM
            fl_clip_rel = max(
                (((a["grads0"][n] * ua_ - fg[n] * uf).norm() / (a["grads0"][n] * ua_).norm().clamp_min(1e-30)).item()
                 for n in a["grads0"]),
                default=0.0,
            )
            fl = (fl_bad, fl_max, fl_rel, fl_clip_rel)
        ga, gb = a["grads0"], b["grads0"]
        keys_ok = set(ga) == set(gb)
        # Undo the (active) clip so the two runs' slightly different coefficients don't count.
        ua = (a["norms"][0] + 1e-6) / MAX_NORM
        ub = (b["norms"][0] + 1e-6) / MAX_NORM
        per = sorted(
            (((ga[n] * ua - gb[n] * ub).norm() / (ga[n] * ua).norm().clamp_min(1e-30)).item(), n) for n in ga
        ) if keys_ok else [(float("inf"), "<key mismatch>")]
        g_rel_max, g_rel_med = per[-1][0], per[len(per) // 2][0]
        norm0_rel = abs(a["norms"][0] - b["norms"][0]) / a["norms"][0]
        norm_rel = max(abs(x - y) / x for x, y in zip(a["norms"], b["norms"]))
        w_diff = max((a["weights"][n].float() - b["weights"][n].float()).abs().max().item() for n in a["weights"])
        clipped = all(n > MAX_NORM for n in a["norms"] + b["norms"])
        mem_ok = b["post_bwd_mib"][-1] < a["post_bwd_mib"][-1]
        bf16_ok = a["grad_dtypes"] == b["grad_dtypes"] == ["torch.bfloat16"]
        checks = {
            (
                "step-1 accumulated pre-clip grads bit-exact"
                if fl is None or fl[0] == 0
                else "step-1 accumulated pre-clip grads rel <= 2x cpu_adam-vs-cpu_adam floor"
            ): pre_keys_ok and (pre_bad == 0 if fl is None or fl[0] == 0 else pre_rel <= 2 * fl[2]),
            "grad keys match": keys_ok,
            "grads bf16 in both": bf16_ok,
            "step-1 clipped grads per-tensor rel <= max(1e-2, 2x floor)": g_rel_max
            <= max(1e-2, 2 * fl[3] if fl else 0.0),
            "step-1 norm rel <= 8e-3 (bf16 eps)": norm0_rel <= 8e-3,
            "all norms rel <= 5e-2": norm_rel <= 5e-2,
            "weights within 2*lr*steps": w_diff <= 2 * LR * STEPS * 1.01,
            "clipping active": clipped,
            "stream holds less GPU memory after backward": mem_ok,
        }
        print(f"--- rank {r} (seq={a['seq']})")
        print(f"step-1 accumulated pre-clip grads stream-vs-cpu_adam: {len(pa)} tensors, {pre_bad} not bit-exact, "
              f"max abs diff {pre_max:.2e}, max per-tensor rel {pre_rel:.2e}")
        if fl:
            print(f"  floor cpu_adam-vs-cpu_adam ({floor}): {fl[0]} not bit-exact, max abs diff {fl[1]:.2e}, "
                  f"max per-tensor rel {fl[2]:.2e}; clipped-grad rel {fl[3]:.2e}")
        print(f"norms cpu_adam: {a['norms']}")
        print(f"norms stream  : {b['norms']}")
        print(f"step-1 clipped grads (clip undone) per-tensor rel diff: max {g_rel_max:.2e} ({per[-1][1]}), median {g_rel_med:.2e}")
        print(f"step-1 norm rel {norm0_rel:.2e}; all-steps norm rel max {norm_rel:.2e}; weights max abs diff {w_diff:.2e}")
        for k in ("peak_mib", "post_bwd_mib", "t_fwd_bwd", "t_opt"):
            print(f"{k:13s} cpu_adam {[round(x, 3) for x in a[k]]}  stream {[round(x, 3) for x in b[k]]}")
        for k, v in checks.items():
            print(f"  [{'ok' if v else 'FAIL'}] {k}")
        ok &= all(checks.values())
    print("STREAM_GRADS_EQUIVALENCE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--mode", choices=["cpu_adam", "stream"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--tag", default=None, help="result file prefix for run (default: the mode)")
    ap.add_argument("--floor", default=None, help="compare: tag of a second cpu_adam run for the noise floor")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.mode, args.out, args.seq, args.tag or args.mode)
    else:
        raise SystemExit(0 if compare(args.out, args.floor) else 1)
