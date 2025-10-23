# src/utils/determinism.py

# Utilities For Reproducible Training And Data Loading
# Provides Global Seed Setup, Torch Generator Creation, And Worker Initialization

# Standard Library
import os
import random

# Third-Party
import numpy as np
import torch

def SetSeed(seed: int, deterministic_cudnn: bool = True):
    # Set Global Random Seed For Python, NumPy, And PyTorch (CPU + GPU)
    if seed is None:
        return  # Skip If No Seed Provided

    os.environ["PYTHONHASHSEED"] = str(seed)        # Fix Hash-Based Operations
    random.seed(seed)                               # Python RNG
    np.random.seed(seed)                            # NumPy RNG
    torch.manual_seed(seed)                         # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)               # PyTorch GPU RNG

    if deterministic_cudnn:
        # Enforce Deterministic CuDNN Paths
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)  # Warn Only If Non-Deterministic

def MakeGenerator(seed: int) -> torch.Generator:
    # Create Torch Generator For DataLoader Or Other Randomized Operations
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def WorkerInitFn(worker_id: int):
    # Initialize DataLoader Worker With Unique Seed
    base = int(os.environ.get("DATA_WORKER_SEED", "0"))
    s = base + worker_id                             # Derive Worker-Specific Seed
    random.seed(s)                                   # Python RNG
    np.random.seed(s & 0xFFFFFFFF)                  # Ensure 32-Bit NumPy Seed
    torch.manual_seed(s)                             # PyTorch RNG
