"""
Reproducibility utilities: set_seed for torch, numpy, random.
Implements TDD §4.2 reproducibility notes.
"""
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value (default: 42)
    
    Sets seed for:
        - Python's built-in random module
        - NumPy
        - PyTorch (CPU and all CUDA devices)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if CUDA available
    
    # For deterministic cuDNN (optional, may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For Python's hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
