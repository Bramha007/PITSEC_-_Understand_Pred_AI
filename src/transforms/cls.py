# src/transforms/cls.py
from typing import Sequence
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

class CLSTransform:
    """
    Train=True: light aug (p=0.5 horizontal flip, random rotation in [-deg,+deg]),
                then resize to `size` and normalize to ImageNet stats.
    Train=False: resize + normalize only.

    Expects input tensors CxHxW in [0,1]. Returns either img or (img, y).
    """
    def __init__(self, size: int = 224, mean: Sequence[float] = _IMAGENET_MEAN,
                 std: Sequence[float] = _IMAGENET_STD, train: bool = False, rot_deg: float = 5.0):
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
        img = F.rotate(
            img, angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False, fill=0
        )
        return img

    def _finalize(self, img: torch.Tensor) -> torch.Tensor:
        # Resize AFTER aug to the network input size
        img = F.resize(img, [self.size, self.size], antialias=True)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __call__(self, *args):
        if len(args) == 1:
            img = args[0]
            if self.train:
                img = self._maybe_flip(img)
                img = self._maybe_rotate(img)
            img = self._finalize(img)
            return img
        elif len(args) == 2:
            img, y = args
            if self.train:
                img = self._maybe_flip(img)
                img = self._maybe_rotate(img)
            img = self._finalize(img)
            return img, y
        else:
            raise TypeError("CLSTransform expects (img) or (img, y)")
