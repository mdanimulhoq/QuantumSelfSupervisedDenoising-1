"""
Device utilities: auto-select CPU/CUDA/MPS.
"""
import multiprocessing
import torch
from typing import Dict, Any, Union, List, TypeVar

T = TypeVar('T')

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Automatically select the best available device.
    
    Priority: CUDA -> MPS (Apple Silicon) -> CPU
    
    Args:
        force_cpu: If True, force CPU even if GPU available
    
    Returns:
        torch.device: Selected device
    """
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Returns:
        dict: Device information including type, count, memory, etc.
    """
    device = get_device()
    info: Dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
    }
    
    if device.type == "cuda":
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated"] = torch.cuda.memory_allocated(0)
        info["memory_reserved"] = torch.cuda.memory_reserved(0)
    elif device.type == "cpu":
        info["cpu_count"] = multiprocessing.cpu_count()
    
    return info

def to_device(data: Union[torch.Tensor, List, Dict, Any], device: torch.device) -> Any:
    """
    Move data to device (handles tensors, lists, dicts).
    
    Args:
        data: Tensor, list, or dict of tensors
        device: Target device
    
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data
