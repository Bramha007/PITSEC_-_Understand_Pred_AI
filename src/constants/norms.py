# src/constants/norms.py

# ImageNet Normalization Constants For Torchvision
# Provides List And Tensor Representations Of Channel Means And Standard Deviations
# Supports Convenient Device Placement For Model Inputs And XAI Computations

from __future__ import annotations

# Standard Library
from typing import List, Tuple

# Third-Party
import torch

# Lists For Torchvision Normalization Transforms
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]  # ImageNet Channel Means (R, G, B)
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]  # ImageNet Channel Standard Deviations (R, G, B)

# Tensors For Model Input Normalization And XAI Computations
IMAGENET_MEAN_T: torch.Tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)  # Reshape For Batch Broadcasting
IMAGENET_STD_T:  torch.Tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)   # Reshape For Batch Broadcasting


# Get ImageNet Normalization Lists (Mean, Std)
def get_norm_lists() -> Tuple[List[float], List[float]]:
    # Returns Mean And Std As Python Lists For Transforms Or Logging
    return IMAGENET_MEAN, IMAGENET_STD


# Get ImageNet Normalization Tensors On A Specific Device
def get_norm_tensors(device: str | torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns Mean And Std Tensors Moved To The Desired Device
    return IMAGENET_MEAN_T.to(device), IMAGENET_STD_T.to(device)
