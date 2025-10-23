# src/transforms/cls.py

# Classification Transforms
# Supports Optional Training Augmentations: Random Flip, Small Rotation
# Always Resizes And Normalizes Using ImageNet Statistics

from typing import Sequence
import random

# Third-Party
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

# Local Modules
from src.constants.norms import get_norm_lists

# ImageNet mean/std for normalization
_IMAGENET_MEAN, _IMAGENET_STD = get_norm_lists()

# Classification Transform With Optional Augmentations
class CLSTransform:
    # train=True: Random Flip + Small Rotation -> Resize -> Normalize
    # train=False: Resize -> Normalize
    # Input: CxHxW tensor in [0,1]; Output: normalized tensor (optionally with label)

    def __init__(
        self,
        size: int = 224,
        mean: Sequence[float] = _IMAGENET_MEAN,
        std: Sequence[float] = _IMAGENET_STD,
        train: bool = False,
        rot_deg: float = 5.0,
    ):
        self.size = int(size)
        self.mean = list(mean)
        self.std = list(std)
        self.train = bool(train)
        self.rot_deg = float(rot_deg)

    # Random horizontal flip
    def _maybe_flip(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            img = F.hflip(img)
        return img

    # Random small rotation
    def _maybe_rotate(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.rot_deg, self.rot_deg)
        return F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)

    # Resize and normalize
    def _finalize(self, img: torch.Tensor) -> torch.Tensor:
        img = F.resize(img, [self.size, self.size], antialias=True)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    # Apply transform to (img) or (img, label)
    def __call__(self, *args):
        if len(args) == 1:
            img = args[0]
            if self.train:
                img = self._maybe_flip(img)
                img = self._maybe_rotate(img)
            return self._finalize(img)

        elif len(args) == 2:
            img, y = args
            if self.train:
                img = self._maybe_flip(img)
                img = self._maybe_rotate(img)
            return self._finalize(img), y

        else:
            raise TypeError("CLSTransform expects (img) or (img, y)")
