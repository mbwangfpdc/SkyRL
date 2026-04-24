import os
from contextlib import contextmanager
from typing import Optional, List
from loguru import logger
import torch
import torch.distributed
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler


class Profiler:
    """
    A PyTorch profiler wrapper class for collecting performance metrics.
    """

    def __init__(self, config):
        """
        config contains:
        - enable: bool
        - ranks: list[int]
        - save_path: str
        """
        self.enable = config.enable
        if not config.enable:
            return
        self.config = config
        self.save_path = config.save_path
        self.ranks = config.ranks
        self.saved = False
        self.prof = None
        self.rank = torch.distributed.get_rank()
        if self.rank in self.ranks:
            logger.info(f"[Profiler] Profiler init for rank {self.rank}")

            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=1,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    def check(self):
        return self.prof is not None and self.enable

    def start(self):
        if self.check():
            logger.info(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            logger.info(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None and not self.saved:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            save_file_name = f"/prof_rank_{self.rank}.json"
            logger.info(f"[Profiler] Saving trace to {self.save_path + save_file_name}")
            self.prof.export_chrome_trace(self.save_path + save_file_name)
            self.enable = False
            self.saved = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        if self.check():
            logger.info(f"[Profiler] Trace stopped for rank {self.rank}")
            self.enable = False


class CudaTimer:
    def __init__(self, device):
        self.device = device

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)  # Calculate the elapsed time

from collections import defaultdict
PROFILE_INDEX_MAP = defaultdict(int)

# TODO: double-check vibecoding
@contextmanager
def profile(
    profile_name: str,
    activities: Optional[List[ProfilerActivity]] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    with_flops: bool = False,
    with_modules: bool = True,
    **kwargs
):
    """
    Context manager for profiling PyTorch operations. Only profiles rank 0 in distributed settings.
    
    Args:
        profile_name: Name of the profile run. Filename generated using this
                    Profiles will be saved to /local_nvme1/mborjigi/output/{profile_name}_{run_index}.json
        activities: List of profiler activities to capture. 
                   Defaults to [ProfilerActivity.CPU, ProfilerActivity.CUDA] if CUDA is available.
        record_shapes: If True, records tensor shapes.
        profile_memory: If True, profiles memory usage.
        with_stack: If True, records source code stack traces.
        with_flops: If True, estimates FLOPs for operations.
        with_modules: If True, records module hierarchy.
        **kwargs: Additional arguments to pass to torch.profiler.profile.
    
    Example:
        ```python
        with profile('forward_pass'):
            model(inputs)
        ```
        
        Or with specific activities and options:
        ```python
        with profile(
            'backward_pass',
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            with_flops=True,
            profile_memory=True
        ):
            for batch in dataloader:
                output = model(batch)
                loss = criterion(output)
                loss.backward()
        ```
    """
    torch_rank = torch.distributed.get_rank()
    output_path = f"/local_nvme1/mborjigi/output/{profile_name}_{PROFILE_INDEX_MAP[profile_name]}.json.{torch_rank}"
    PROFILE_INDEX_MAP[profile_name] += 1
    import psutil
    logger.info(f"Profiling process {psutil.Process(os.getpid()).name()} (PID: {os.getpid()})")
    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
    
    profiler_kwargs = {
        'activities': activities,
        'record_shapes': record_shapes,
        'profile_memory': profile_memory,
        'with_stack': with_stack,
        'with_flops': with_flops,
        'with_modules': with_modules,
        **kwargs
    }
    
    profiler = torch.profiler.profile(**profiler_kwargs)
    
    try:
        profiler.__enter__()
        yield profiler
    finally:
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(output_path)

