# src/transforms/det_experimental.py

# Experimental Detection Transforms (Not Used In Main Pipeline)
# API: (Img, Target) -> (Img, Target), Same As det.py

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
import random
from typing import Tuple

# Third-Party
import torch
from torchvision.transforms import functional as F

__all__ = [
    "AnisotropicScale",
    "RandomResize",
]

def _GetSize(img) -> Tuple[int, int]:
    # Return (Width, Height) Of PIL Or Tensor Image
    w, h = F.get_image_size(img)
    return int(w), int(h)

@dataclass
class AnisotropicScale:
    # Independently Scale Width And Height By Random Factors [MinScale, MaxScale]
    min_scale: float = 0.8
    max_scale: float = 1.3
    antialias: bool = True

    def __call__(self, img, target):
        sx, sy = random.uniform(self.min_scale, self.max_scale), random.uniform(self.min_scale, self.max_scale)
        w, h = _GetSize(img)
        new_w, new_h = max(1, int(round(w * sx))), max(1, int(round(h * sy)))

        img = F.resize(img, [new_h, new_w], antialias=self.antialias)

        # Scale Ground Truth Boxes If Present
        if target is not None and "boxes" in target and target["boxes"] is not None:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= (new_w / max(w, 1))
            boxes[:, [1, 3]] *= (new_h / max(h, 1))
            target = {**target, "boxes": boxes}

        return img, target

@dataclass
class RandomResize:
    # Resize To Random Short Side; Clamp Long Side If Needed
    short_side_choices: tuple[int, ...] = (320, 400, 480, 560, 640)
    max_long_side: int | None = 1024
    antialias: bool = True

    def __call__(self, img, target):
        short_side = random.choice(self.short_side_choices)
        w, h = _GetSize(img)
        if min(w, h) == 0:
            return img, target

        scale = short_side / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        # Clamp If Longest Side Exceeds Max
        if self.max_long_side is not None and max(new_w, new_h) > self.max_long_side:
            clamp_scale = self.max_long_side / max(new_w, new_h)
            new_w, new_h = int(round(new_w * clamp_scale)), int(round(new_h * clamp_scale))

        img = F.resize(img, [new_h, new_w], antialias=self.antialias)

        # Scale Ground Truth Boxes If Present
        if target is not None and "boxes" in target and target["boxes"] is not None:
            sx, sy = new_w / max(w, 1), new_h / max(h, 1)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target = {**target, "boxes": boxes}

        return img, target
