import copy
import gc
import json
import os
import random
import time
from collections import defaultdict
from datetime import timedelta
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from loguru import logger
from packaging import version
from torch import distributed as dist
from torch import optim
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.trainer import get_scheduler

from skyrl.train.utils import (
    Timer,
    time_func,
)
from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    PrecisionType,
    apply_fsdp2,
    create_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_get_full_state_dict,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_state_ctx,
    get_fsdp_wrap_policy,
    get_sharding_strategy,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.backends.skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import FSDPConfig, ModelConfig, OptimizerConfig

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
    )
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
    )
else:
    CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy = None, None, None


class FSDPStrategy(DistributedStrategy):
    """
    The strategy for training with FSDP.
    """

    def __init__(
        self,
        fsdp_config: FSDPConfig,
        optimizer_config: Optional[OptimizerConfig] = None,
        model_config: Optional[ModelConfig] = None,
        fsdp_strategy: str = "fsdp",
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        num_training_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert fsdp_strategy in ("fsdp", "fsdp2"), f"Unsupported FSDP strategy: {fsdp_strategy}"
        self.fsdp_config = fsdp_config
        self.optimizer_config = optimizer_config
        self.model_config = model_config
        self.fsdp_strategy = fsdp_strategy
        self.max_norm = optimizer_config.max_grad_norm if optimizer_config is not None else 1.0
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.seed = seed
        self.device_mesh = None
        self.total_training_steps: Optional[int] = num_training_steps

        # if we are using fsdp 1 or cpu offload is off for fsdp2, then we need to manually offload weights/optimizer to cpu
        self.manual_offload = self.fsdp_strategy == "fsdp" or not self.fsdp_config.cpu_offload
        self.cpu_adam = bool(self.optimizer_config is not None and self.optimizer_config.cpu_adam)
        if self.cpu_adam and self.fsdp_strategy != "fsdp2":
            raise NotImplementedError(
                "optimizer_config.cpu_adam=True is only implemented for fsdp_strategy='fsdp2' -- it "
                "relies on FSDP2's per-parameter DTensor sharding to clone/writeback named CPU "
                "masters; FSDP1's flat-parameter sharding (use_orig_params=False) is incompatible "
                "with that."
            )
        if self.cpu_adam and self.fsdp_config.cpu_offload:
            raise ValueError(
                "optimizer_config.cpu_adam=True is incompatible with fsdp_config.cpu_offload=True: "
                "both keep a full CPU-resident copy of the model/optimizer via different "
                "mechanisms, so stacking them only doubles CPU memory for no benefit. Pick one."
            )
        if self.optimizer_config is not None:
            # cpu_adam keeps optimizer state on CPU by construction (see _fsdp_init_train_model /
            # optimizer_step) -- offload_after_step's GPU<->CPU round trip would be pure overhead
            # on top of that, so cpu_adam takes precedence rather than stacking with it.
            self.manual_offload_optimizer = (
                self.optimizer_config.offload_after_step and self.manual_offload and not self.cpu_adam
            )
        else:
            self.manual_offload_optimizer = False

        # LoRA related configs
        self.is_lora = self.model_config.lora.rank > 0 if self.model_config is not None else False
        if self.cpu_adam and self.is_lora:
            raise NotImplementedError("optimizer_config.cpu_adam=True is not yet supported with LoRA.")

        # cpu_adam: per-role dict of CPU fp32 master nn.Parameters, keyed by the FSDP module's
        # named_parameters() name. Populated in _fsdp_init_train_model; the optimizer is built
        # directly over these instead of the GPU-resident FSDP2 shards. See optimizer_step for the
        # per-step grad-copy-down / step-on-CPU / value-copy-back sequence.
        self._cpu_master: dict = {}

        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        self.world_size = dist.get_world_size()

        # Log initial GPU memory state

        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=self.fsdp_config.fsdp_size)

    @time_func("FSDPStrategy.offload_to_cpu")
    def offload_to_cpu(
        self, model, optimizer, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True
    ):
        """
        Offload model weights and optimizer to CPU memory.

        For all cases except fsdp2 with cpu_offload=True, we need to manually offload weights/optimizer to cpu.
        """

        if isinstance(model, HFModelWrapper):
            model = model.model
        else:
            model = model

        if self.manual_offload:
            if offload_model:
                logger.info(f"Offloading model to CPU (manual_offload={self.manual_offload})...")
                offload_fsdp_model_to_cpu(model, empty_cache=True)

            if optimizer is not None and self.manual_offload_optimizer and offload_optimizer:
                logger.info(f"Offloading optimizer to CPU...")
                offload_fsdp_optimizer(optimizer)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @time_func("FSDPStrategy.backload_to_gpu")
    def backload_to_gpu(self, model, optimizer, non_blocking=True, backload_optimizer=True, backload_model=True):
        """Reload model weights back to GPU."""

        if isinstance(model, HFModelWrapper):
            model = model.model
        else:
            model = model

        # if we are using fsdp 1 or cpu offload is off for fsdp2, then we need to manually backload weights/optimizer to gpu
        if self.manual_offload:
            if backload_model:
                logger.info(f"Backloading model to GPU (manual_offload={self.manual_offload})...")
                load_fsdp_model_to_gpu(model)
            if optimizer is not None and self.manual_offload_optimizer and backload_optimizer:
                logger.info(f"Backloading optimizer to GPU...")
                load_fsdp_optimizer(optimizer, torch.cuda.current_device())

        torch.cuda.synchronize()

    @time_func("FSDPStrategy.backward")
    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        """Perform backward pass"""
        loss.backward()

    @time_func("FSDPStrategy.optimizer_step")
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        grad_norm = None
        if isinstance(model, HFModelWrapper):
            model = model.model

        if self.max_norm > 0:
            # NOTE (sumanthrh): All `grad_norm`s returned here are the original grad norms before clipping.
            if isinstance(model, FSDP):
                grad_norm = model.clip_grad_norm_(max_norm=self.max_norm)
            elif isinstance(model, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

        # Skip update if gradient norm is not finite
        if grad_norm is not None and not torch.isfinite(grad_norm):
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                logger.warning(f"rank {rank} grad_norm is not finite: {grad_norm}")
            else:
                logger.warning(f"grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            if self.cpu_adam:
                # optimizer.zero_grad() above only clears the CPU masters' .grad (they are what
                # `optimizer` was built over); the GPU model's own .grad was never touched on this
                # skip path, so clear it too or the next backward's accumulation would be wrong.
                for _, param in model.named_parameters():
                    param.grad = None
            return grad_norm

        if self.cpu_adam:
            # grad_norm above is already the value to report (computed pre-clip on the GPU model,
            # same as the non-cpu_adam path); the CPU step itself doesn't produce a new one.
            self._cpu_adam_step(model, optimizer, scheduler)
        else:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        return grad_norm

    def _cpu_adam_step(self, model, optimizer, scheduler):
        """cpu_adam optimizer step: grad clipping has already run on the GPU model above.

        Copies grad shards GPU->CPU into the persistent masters, steps AdamW on CPU, copies the
        updated values back into the live GPU shards in place. Optimizer *state*
        (exp_avg/exp_avg_sq) never leaves CPU -- only this one-shot grad-down / weight-up copy
        crosses PCIe, unlike offload_after_step's round trip of the full optimizer state every step.
        """
        # --- perf breakdown instrumentation (diagnostic; not part of the metrics pipeline) ---
        # Mirrors the [perf-breakdown] style in worker.py's forward_backward. Confirms the cost is
        # actually where the design intends: grad_copy/writeback bounded by the one-shot PCIe
        # transfer of the (much smaller than 2x-state) grad/weight tensors, step_cpu dominated by
        # CPU-bound AdamW and roughly independent of anything GPU-side.
        _t0 = time.time()
        device = torch.cuda.current_device()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            local_grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            self._cpu_master[name].grad = local_grad.detach().to(
                device="cpu", dtype=self._cpu_master[name].dtype, non_blocking=True,
            )
        torch.cuda.synchronize()
        for _, param in model.named_parameters():
            param.grad = None
        _t1 = time.time()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        _t2 = time.time()

        with torch.no_grad():
            for name, param in model.named_parameters():
                local = param.to_local() if isinstance(param, DTensor) else param.data
                local.copy_(self._cpu_master[name].to(device, dtype=local.dtype), non_blocking=True)
        torch.cuda.synchronize()
        _t3 = time.time()
        logger.opt(depth=1).info(
            "[cpu-adam-breakdown] rank={rank}: grad_copy_gpu2cpu={gc:.3f}s step_cpu={st:.3f}s "
            "writeback_cpu2gpu={wb:.3f}s total={tot:.3f}s".format(
                rank=self.get_rank(), gc=_t1 - _t0, st=_t2 - _t1, wb=_t3 - _t2, tot=_t3 - _t0,
            )
        )

    @time_func("FSDPStrategy.prepare")
    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models and optimizers with FSDP"""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._fsdp_init_train_model(*arg))
            else:
                ret.append(self._fsdp_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    @time_func("FSDPStrategy._fsdp_init_model")
    def _fsdp_init_model(self, model, is_train=True, is_wrapped=False) -> FSDP:
        logger.info(f"FSDP config is: {self.fsdp_config}")
        # Initialize FSDP wrapping policy
        wrap_policy = get_fsdp_wrap_policy(
            module=model.model if is_wrapped else model,
            config=getattr(self.fsdp_config, "wrap_policy", None),
            is_lora=self.is_lora,
        )

        # Setup mixed precision
        mixed_precision_config = getattr(self.fsdp_config, "mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.param_dtype)
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.reduce_dtype)
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.buffer_dtype)
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        cpu_offload = None

        # sharding strategy
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Wrap model with FSDP
        if self.fsdp_strategy == "fsdp":
            # cpu offloading will always be none for models that train with FSDP due to correctness issues with gradient accumulation -
            # see https://docs.pytorch.org/docs/stable/fsdp.html
            if not is_train and self.fsdp_config.cpu_offload:
                cpu_offload = CPUOffload(offload_params=True)
            fsdp_module = FSDP(
                model.model if is_wrapped else model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif self.fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if self.fsdp_config.cpu_offload:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": self.fsdp_config.reshard_after_forward,
            }
            module = model.model if is_wrapped else model
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.fsdp_config)
            fsdp2_load_full_state_dict(module, full_state, cpu_offload)
            fsdp_module = module
        else:
            raise NotImplementedError(f"{self.fsdp_strategy} not implemented")

        return fsdp_module

    def _init_cpu_adam_masters(self, fsdp_module) -> list:
        """Build persistent CPU master weights for cpu_adam and return the optimizer's param list.

        One fp32 (``optimizer_config.master_dtype``) CPU clone per named parameter's local shard.
        The FSDP2 model is left untouched here -- it stays GPU-resident and is what forward/backward
        run against; optimizer_step() copies grads into these masters, steps AdamW on them, then
        copies the updated values back into the GPU shards in place. Unlike granular's
        dynamic_groups mode, there is no shared-memory store: SkyRL never rebuilds the FSDP group
        mid-run, so plain in-process tensors are enough, and they round-trip through the existing
        per-rank optimizer.state_dict() checkpoint path unchanged.
        """
        master_dtype = PrecisionType.to_dtype(self.optimizer_config.master_dtype)
        # Ray pins each actor to a single CPU thread by default; AdamW's foreach update is
        # embarrassingly parallel and memory-bandwidth-bound, so that leaves most of the box idle.
        # Mirrors granular-cais-rl's cpu_adam thread tuning (~5.5x measured at 32 threads/rank).
        n_threads = int(os.environ.get("SKYRL_CPU_ADAM_THREADS", "0")) or max(
            8, min(32, (os.cpu_count() or 16) // max(1, 2 * self.world_size))
        )
        torch.set_num_threads(n_threads)
        logger.info(
            f"[cpu_adam] rank={self.get_rank()}: torch intra-op CPU threads -> {n_threads} "
            f"(cpu_count={os.cpu_count()}, world_size={self.world_size}, master_dtype={master_dtype})"
        )
        self._cpu_master = {}
        for name, param in fsdp_module.named_parameters():
            local = param.to_local() if isinstance(param, DTensor) else param
            self._cpu_master[name] = nn.Parameter(
                local.detach().to(device="cpu", dtype=master_dtype, copy=True)
            )
        return list(self._cpu_master.values())

    @time_func("FSDPStrategy._fsdp_init_train_model")
    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        """Initialize a model for training with FSDP"""
        is_wrapped = isinstance(model, HFModelWrapper)
        fsdp_module = self._fsdp_init_model(model, is_train=True, is_wrapped=is_wrapped)

        optim_config = self.optimizer_config
        if optim_config is not None:
            opt_params = (
                self._init_cpu_adam_masters(fsdp_module) if self.cpu_adam else fsdp_module.parameters()
            )
            new_optimizer = optim.AdamW(
                opt_params,
                lr=optim_config.lr,
                betas=optim_config.adam_betas,
                weight_decay=optim_config.weight_decay,
            )
            # try:
            #     # The standard v0.9 path
            #     from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            # except ImportError:
            #     # Sometimes it wasn't exposed in the __init__, so try the file directly
            #     from torchao.prototype.low_bit_optim.cpu_offload import CPUOffloadOptimizer
            # new_optimizer = CPUOffloadOptimizer(
            #     fsdp_module.parameters(),
            #     optimizer_class=optim.AdamW,
            #     lr=optim_config.lr,
            #     betas=optim_config.adam_betas,
            #     weight_decay=optim_config.weight_decay,
            #     # TODO: debugging this
            #     offload_gradients=True
            # )

            lr_scheduler = get_scheduler(
                optim_config.scheduler,
                new_optimizer,
                num_warmup_steps=optim_config.num_warmup_steps,
                num_training_steps=self.total_training_steps,
            )
        else:
            new_optimizer = None
            lr_scheduler = None

        if is_wrapped:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model, new_optimizer, lr_scheduler

    def _fsdp_init_eval_model(self, model):
        """Initialize a model for evaluation with FSDP"""
        is_wrapped = isinstance(model, HFModelWrapper)
        fsdp_module = self._fsdp_init_model(model, is_train=False, is_wrapped=is_wrapped)

        if is_wrapped:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap model from HFModelWrapper or FSDP"""
        # Handle HFModelWrapper wrapper
        if isinstance(model, HFModelWrapper):
            return self._unwrap_model(model.model)

        # For FSDP2 models, check if the FSDP model itself has the necessary attributes
        model_type = type(model).__name__
        if "FSDP" in model_type:
            has_config = hasattr(model, "config")
            has_lm_head = hasattr(model, "lm_head")
            has_generate = hasattr(model, "generate")
            if has_config and (has_lm_head or has_generate):
                return model

        # Check for FSDP v1 unwrapping
        if hasattr(model, "_fsdp_wrapped_module"):
            return model._fsdp_wrapped_module

        # If no unwrapping needed, return the original model
        return model

    def _fix_fsdp_config(self, config):
        """Fix architecture names by removing FSDP prefix if present"""
        # Determine which config to save
        config_to_save = config

        # Fix architecture name by removing FSDP prefix if present
        if hasattr(config_to_save, "architectures") and config_to_save.architectures:
            # Create a copy of the config to avoid modifying the original
            config_to_save = copy.deepcopy(config_to_save)

            # Fix architecture names to remove FSDP prefix
            fixed_architectures = []
            for arch in config_to_save.architectures:
                fixed_arch = arch
                if arch.startswith("FSDP"):
                    # Remove "FSDP" prefix (for fsdp2)
                    fixed_arch = arch[len("FSDP") :]
                    self.print(f"[rank-0]: Fixed architecture name: {arch} -> {fixed_arch}")
                fixed_architectures.append(fixed_arch)

            config_to_save.architectures = fixed_architectures

        return config_to_save

    def _save_lora_adapters(self, model, ckpt_dir):
        """Save LoRA adapters in HuggingFace PEFT format"""
        from dataclasses import asdict

        from safetensors.torch import save_file

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            layered_summon_lora_params,
        )

        lora_save_path = os.path.join(ckpt_dir, "lora_adapter")
        peft_config = {}

        if self.is_rank_0():
            io.makedirs(lora_save_path, exist_ok=True)
            peft_config = asdict(model.peft_config.get("default", {}))
            if peft_config:
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_params = layered_summon_lora_params(model)

        if self.is_rank_0():
            save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
            with io.open_file(os.path.join(lora_save_path, "adapter_config.json"), "w") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            self.print(f"[rank-0]: Saved LoRA adapter to: {lora_save_path}")

        dist.barrier()

    @time_func("FSDPStrategy.save_checkpoint")
    def save_checkpoint(
        self,
        model,
        ckpt_dir,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        """Save model checkpoint for FSDP"""
        import warnings

        from torch.distributed.fsdp import (
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # Wait for checkpoint directory to be created.
        dist.barrier()

        # Extract the actual model for saving
        if isinstance(model, HFModelWrapper):
            save_model = model.model
        else:
            save_model = model

        if self.fsdp_strategy not in ("fsdp", "fsdp2"):
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Set up state dict configurations for sharded saving
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Define paths for saving individual rank files
        rank = self.get_rank()
        world_size = self.world_size

        with io.local_work_dir(ckpt_dir) as work_dir:
            model_path = os.path.join(work_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
            optim_path = os.path.join(work_dir, f"optim_world_size_{world_size}_rank_{rank}.pt")
            extra_path = os.path.join(work_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt")

            # Save using appropriate FSDP context
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with get_fsdp_state_ctx(save_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                    # Get and save model state dict
                    model_state_dict = save_model.state_dict()
                    self.print(f"[rank-{rank}]: Saving model to {model_path}")
                    with io.open_file(model_path, "wb") as f:
                        torch.save(model_state_dict, f)

                    # Get and save optimizer state dict if optimizer is provided
                    optimizer_state_dict = {}
                    if optimizer is not None:
                        optimizer_state_dict = optimizer.state_dict()
                    self.print(f"[rank-{rank}]: Saving optim to {optim_path}")
                    with io.open_file(optim_path, "wb") as f:
                        torch.save(optimizer_state_dict, f)

                    # Get scheduler state dict if scheduler is provided
                    lr_scheduler_state_dict = {}
                    if scheduler is not None:
                        lr_scheduler_state_dict = scheduler.state_dict()

                    # Create extra state dict with client state and any additional info
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "client_state": client_state,
                        "tag": tag,
                        "fsdp_strategy": self.fsdp_strategy,
                        "world_size": world_size,
                        "rank": rank,
                        "rng": self.get_rng_state(),  # Add RNG state for reproducibility
                    }

                    # Save extra state
                    self.print(f"[rank-{rank}]: Saving extra_state to {extra_path}")
                    with io.open_file(extra_path, "wb") as f:
                        torch.save(extra_state_dict, f)

                    # Garbage collect temporary buffers from materializing the state dicts
                    gc.collect()

            if self.is_rank_0():
                config_save_model = self._unwrap_model(model)
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(config_save_model.config, hf_dir, tokenizer)

                # Also save runtime FSDP config
                fsdp_config_path = os.path.join(work_dir, "fsdp_config.json")
                with io.open_file(fsdp_config_path, "w") as f:
                    json.dump({"fsdp_strategy": self.fsdp_strategy, "world_size": self.world_size}, f, indent=4)

        # Save LoRA adapters if using LoRA
        if self.is_lora and hasattr(save_model, "peft_config"):
            self._save_lora_adapters(save_model, ckpt_dir)

        # Final barrier to ensure all operations complete
        dist.barrier()
        torch.cuda.synchronize()
        self.print(f"[rank-{rank}]: Checkpoint saved to {ckpt_dir}")

    @time_func("FSDPStrategy.load_checkpoint")
    def load_checkpoint(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    ):
        """Load model checkpoint for FSDP"""
        import warnings

        from torch.distributed.fsdp import (
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        if ckpt_dir is None:
            raise ValueError("ckpt_dir cannot be None")
        elif not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract the actual model for loading
        load_model = model
        if isinstance(model, HFModelWrapper):
            load_model = model.model

        # Define paths for loading individual rank files
        rank = self.get_rank()
        world_size = self.world_size

        with io.local_read_dir(ckpt_dir) as read_dir:
            model_path = os.path.join(read_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
            optim_path = os.path.join(read_dir, f"optim_world_size_{world_size}_rank_{rank}.pt")
            extra_path = os.path.join(read_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt")

            # Check if checkpoint files exist
            if not io.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            if not io.exists(extra_path):
                raise FileNotFoundError(f"Extra state checkpoint not found: {extra_path}")

            # Optimizer path is optional since we may not save optimizer states initially
            optim_exists = io.exists(optim_path)

            self.print(f"[rank-{rank}]: Loading model from {model_path}")
            self.print(f"[rank-{rank}]: Loading extra_state from {extra_path}")
            if optim_exists:
                self.print(f"[rank-{rank}]: Loading optim from {optim_path}")

            # Load state dictionaries from disk
            with io.open_file(model_path, "rb") as f:
                model_state_dict = torch.load(f, map_location="cpu", weights_only=False)
            with io.open_file(extra_path, "rb") as f:
                extra_state_dict = torch.load(f, map_location="cpu", weights_only=False)

            optimizer_state_dict = {}
            if optim_exists and load_optimizer_states:
                with io.open_file(optim_path, "rb") as f:
                    optimizer_state_dict = torch.load(f, map_location="cpu", weights_only=False)

        # Extract scheduler state from extra state
        lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler", {})

        # Set up state dict configurations for sharded loading
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Load using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(load_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                # Load model state dict
                load_model.load_state_dict(model_state_dict, strict=load_module_strict)
                self.print(f"[rank-{rank}]: Successfully loaded model state dict")

                # cpu_adam: the masters were cloned from the GPU model's weights at construction time
                # (init_model), before this checkpoint's weights were loaded into it above -- refresh
                # them from the now-current GPU values so the next optimizer.step() resumes from the
                # checkpoint's weights, not the pre-load ones. In-place so the Parameter identity
                # (what optimizer.load_state_dict below keys exp_avg/exp_avg_sq by) is unchanged.
                if self.cpu_adam and self._cpu_master:
                    with torch.no_grad():
                        for name, param in load_model.named_parameters():
                            local = param.to_local() if isinstance(param, DTensor) else param.data
                            self._cpu_master[name].data.copy_(local.to("cpu", dtype=self._cpu_master[name].dtype))

                # Load optimizer state dict if optimizer object is provided and loading is requested
                if optimizer is not None and load_optimizer_states and optimizer_state_dict:
                    optimizer.load_state_dict(optimizer_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded optimizer state")

                # Load scheduler state dict if scheduler object is provided and loading is requested
                if scheduler is not None and load_lr_scheduler_states:
                    scheduler.load_state_dict(lr_scheduler_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded scheduler state")

        # Load RNG state for reproducibility
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        # Wait for all ranks to finish loading
        dist.barrier()

        # Create states dict with extra information
        client_state = extra_state_dict.get("client_state", {})
        states = {
            "client_state": client_state,
            "tag": extra_state_dict.get("tag", tag),
            "fsdp_strategy": extra_state_dict.get("fsdp_strategy", self.fsdp_strategy),
            "world_size": extra_state_dict.get("world_size", world_size),
            "rank": extra_state_dict.get("rank", rank),
        }

        self.print(f"[rank-{rank}]: Checkpoint loaded successfully from {ckpt_dir}")

        return ckpt_dir, states

    # TODO (erictang000): Test in multi-node setting
    @time_func("FSDPStrategy.save_hf_model")
    def save_hf_model(self, model: Union[HFModelWrapper, nn.Module], output_dir: str, tokenizer=None, **kwargs) -> None:
        """Save model in HuggingFace safetensors format using FSDP's full state dict gathering"""

        # Step 1: Create output directory (rank 0 only)
        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
            self.print(f"[rank-0]: Created output directory: {output_dir}")

        # Step 2: Extract models - get both the model for saving metadata and the FSDP model for state dict
        model_to_save = self._unwrap_model(model)  # For saving config/metadata
        fsdp_model = model.model if isinstance(model, HFModelWrapper) else model  # For state dict collection

        # Validate that we have a proper HuggingFace model
        if not hasattr(model_to_save, "config") or not hasattr(model_to_save, "save_pretrained"):
            raise ValueError("Model must be a HuggingFace model with config and save_pretrained method")

        # Step 3: Determine FSDP version and collect full state dict
        fsdp_ver = fsdp_version(fsdp_model)
        self.print(f"[rank-{self.get_rank()}]: Detected FSDP version: {fsdp_ver}")

        if fsdp_ver == 2:
            # Use FSDP2 API - collects on rank 0 only
            output_state_dict = fsdp2_get_full_state_dict(fsdp_model, cpu_offload=True, rank0_only=True)
        elif fsdp_ver == 1:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False)
            output_state_dict = get_model_state_dict(fsdp_model, options=options)
            if not self.is_rank_0():
                output_state_dict.clear()
        else:
            raise ValueError(f"Unsupported FSDP version: {fsdp_ver}")

        # Step 4: Save on rank 0 only
        if self.is_rank_0():
            with io.local_work_dir(output_dir) as work_dir:
                # Save the model in HuggingFace format using safetensors
                model_to_save.save_pretrained(work_dir, state_dict=output_state_dict, safe_serialization=True, **kwargs)

                # Fix and save the config
                config_to_save = self._fix_fsdp_config(model_to_save.config)
                config_to_save.save_pretrained(work_dir)

                # Save tokenizer if provided
                if tokenizer is not None:
                    tokenizer.save_pretrained(work_dir)

            self.print(f"[rank-0]: Successfully saved model to {output_dir}")

        dist.barrier()
