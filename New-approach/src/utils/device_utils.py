# File: src/utils/device_utils.py

import torch
import warnings

def select_device(user_device: str = "auto") -> torch.device:
    """
    Selects the best available device (CUDA > CPU) based on user configuration.
    
    Args:
        user_device: String specifying the desired device ('auto', 'cpu', or 'cuda').
    
    Returns:
        A torch.device object.
    """
    device_name = user_device.lower()
    
    if (device_name == 'auto' or device_name == 'cuda') and torch.cuda.is_available():
        # 1. CUDA (NVIDIA GPU) is the fastest option
        device = torch.device('cuda')
    elif device_name == 'auto':
        # 2. Default to CPU if auto is selected and CUDA is not available
        device = torch.device('cpu')
        if user_device != 'cpu':
             warnings.warn("CUDA device not available. Falling back to CPU.", UserWarning)
    elif device_name == 'cpu':
        # 3. Explicitly requested CPU
        device = torch.device('cpu')
    elif device_name == 'cuda' and not torch.cuda.is_available():
        # 4. CUDA requested but not available
        device = torch.device('cpu')
        warnings.warn("CUDA device explicitly requested but not available. Using CPU.", UserWarning)
    else:
        # Fallback for unexpected string, default to CPU
        device = torch.device('cpu')

    print(f"Using device: {device.type.upper()}")
    return device