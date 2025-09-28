# src/constants/norms.py
from __future__ import annotations
from typing import List, Tuple
import torch

# Lists (for torchvision transforms)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

# Tensors (for model/XAI code)
IMAGENET_MEAN_T: torch.Tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
IMAGENET_STD_T:  torch.Tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

def get_norm_lists() -> Tuple[List[float], List[float]]:
    return IMAGENET_MEAN, IMAGENET_STD

def get_norm_tensors(device: str | torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    return IMAGENET_MEAN_T.to(device), IMAGENET_STD_T.to(device)
