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
