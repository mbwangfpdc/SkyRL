"""GPU memory tracking utilities for training diagnostics."""

import torch
from loguru import logger


def get_gpu_memory_stats(device_id: int = None) -> dict:
    """Get comprehensive GPU memory statistics.
    
    Args:
        device_id: GPU device ID (defaults to current device)
        
    Returns:
        Dict with keys: allocated_mb, reserved_mb, free_mb, total_mb, utilization_pct, max_allocated_mb
    """
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    torch.cuda.synchronize(device_id)
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    max_allocated = torch.cuda.max_memory_allocated(device_id)
    
    allocated_mb = allocated / 1024 / 1024
    reserved_mb = reserved / 1024 / 1024
    free_mb = free / 1024 / 1024
    total_mb = total / 1024 / 1024
    utilization_pct = (allocated / total) * 100 if total > 0 else 0
    max_allocated_mb = max_allocated / 1024 / 1024
    
    return {
        "allocated_mb": allocated_mb,
        "reserved_mb": reserved_mb,
        "free_mb": free_mb,
        "total_mb": total_mb,
        "utilization_pct": utilization_pct,
        "max_allocated_mb": max_allocated_mb,
    }


def log_gpu_memory(tag: str, level: str = "info", device_id: int = None) -> dict:
    """Log GPU memory statistics with a descriptive tag.
    
    Args:
        tag: Description of the checkpoint (e.g., "before_forward", "after_backward")
        level: Log level ("debug", "info", "warning")
        device_id: GPU device ID (defaults to current device)
        
    Returns:
        The stats dict for later comparison (used by log_gpu_memory_delta)
    """
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    stats = get_gpu_memory_stats(device_id)
    
    msg = (
        f"[GPU Memory {tag}] "
        f"Allocated: {stats['allocated_mb']:.1f}MB / {stats['reserved_mb']:.1f}MB reserved | "
        f"Utilization: {stats['utilization_pct']:.1f}% | "
        f"Free: {stats['free_mb']:.1f}MB / {stats['total_mb']:.1f}MB total | "
        f"Peak: {stats['max_allocated_mb']:.1f}MB"
    )
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg)
    
    return stats


def log_gpu_memory_delta(tag: str, start_stats: dict, level: str = "info", device_id: int = None) -> dict:
    """Log memory changes from a previous checkpoint.
    
    Args:
        tag: Description of the operation (e.g., "forward_pass", "backward_pass")
        start_stats: Previous stats dict from log_gpu_memory()
        level: Log level ("debug", "info", "warning")
        device_id: GPU device ID (defaults to current device)
        
    Returns:
        The current stats dict
    """
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    current_stats = get_gpu_memory_stats(device_id)
    
    delta_allocated = current_stats["allocated_mb"] - start_stats["allocated_mb"]
    delta_max_allocated = current_stats["max_allocated_mb"] - start_stats["max_allocated_mb"]
    
    sign_allocated = "+" if delta_allocated >= 0 else ""
    sign_max = "+" if delta_max_allocated >= 0 else ""
    
    msg = (
        f"[GPU Memory Delta {tag}] "
        f"Δ Allocated: {sign_allocated}{delta_allocated:.1f}MB → {current_stats['allocated_mb']:.1f}MB | "
        f"Δ Peak: {sign_max}{delta_max_allocated:.1f}MB → {current_stats['max_allocated_mb']:.1f}MB | "
        f"Free: {current_stats['free_mb']:.1f}MB | "
        f"Util: {current_stats['utilization_pct']:.1f}%"
    )
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg)
    
    return current_stats


def estimate_model_size(model) -> dict:
    """Estimate total model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with keys: params_mb, buffers_mb, total_mb
    """
    params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    buffers_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    total_mb = params_mb + buffers_mb
    
    msg = f"[Model Size] Params: {params_mb:.1f}MB | Buffers: {buffers_mb:.1f}MB | Total: {total_mb:.1f}MB"
    logger.info(msg)
    
    return {
        "params_mb": params_mb,
        "buffers_mb": buffers_mb,
        "total_mb": total_mb,
    }


def check_memory_pressure(threshold_pct: float = 90.0, device_id: int = None) -> bool:
    """Check if GPU memory utilization exceeds threshold and log warning.
    
    Args:
        threshold_pct: Utilization threshold percentage (default 90%)
        device_id: GPU device ID (defaults to current device)
        
    Returns:
        True if threshold exceeded, False otherwise
    """
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    stats = get_gpu_memory_stats(device_id)
    
    if stats["utilization_pct"] >= threshold_pct:
        logger.warning(
            f"[Memory Pressure Alert] GPU {device_id} utilization {stats['utilization_pct']:.1f}% "
            f"exceeds threshold {threshold_pct}% "
            f"(Allocated: {stats['allocated_mb']:.1f}MB / {stats['total_mb']:.1f}MB)"
        )
        return True
    
    return False
