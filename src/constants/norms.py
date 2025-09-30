# src/constants/norms.py
# Provides ImageNet Mean And Standard Deviation For Normalization
# Includes Lists And Torch Tensors For Training, Evaluation, And XAI

from __future__ import annotations
from typing import List, Tuple
import torch

# Lists For Torchvision Transforms
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

# Tensors For Model And XAI Code
IMAGENET_MEAN_T: torch.Tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
IMAGENET_STD_T:  torch.Tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


# Get Lists (Mean, Std)
def get_norm_lists() -> Tuple[List[float], List[float]]:
    return IMAGENET_MEAN, IMAGENET_STD


# Get Tensors (Mean, Std) On A Device
def get_norm_tensors(device: str | torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    return IMAGENET_MEAN_T.to(device), IMAGENET_STD_T.to(device)
