# src/transforms/cls.py

# Classification Transforms With ImageNet Normalization And Optional Augmentation

from typing import Sequence
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from src.constants.norms import get_norm_lists

# ImageNet mean/std for normalization
_IMAGENET_MEAN, _IMAGENET_STD = get_norm_lists()


class CLSTransform:
    # Train=True -> Random Flip + Small Rotation -> Resize -> Normalize
    # Train=False -> Resize -> Normalize
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

    def _maybe_flip(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            img = F.hflip(img)
        return img

    def _maybe_rotate(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.rot_deg, self.rot_deg)
        return F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)

    def _finalize(self, img: torch.Tensor) -> torch.Tensor:
        img = F.resize(img, [self.size, self.size], antialias=True)  # resize after aug
        img = F.normalize(img, mean=self.mean, std=self.std)         # normalize to ImageNet stats
        return img

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
