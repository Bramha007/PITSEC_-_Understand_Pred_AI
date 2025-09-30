# src/transforms/det_experimental.py
# Experimental detection transforms (not used in main pipeline)
# API: (img, target) -> (img, target), like det.py

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Tuple
import torch
from torchvision.transforms import functional as F

__all__ = [
    "AnisotropicScale",
    "RandomResize",
]


def _get_size(img) -> Tuple[int, int]:
    w, h = F.get_image_size(img)
    return int(w), int(h)


@dataclass
class AnisotropicScale:
    # Independently scale width and height by random factors in [min_scale, max_scale]
    min_scale: float = 0.8
    max_scale: float = 1.3
    antialias: bool = True

    def __call__(self, img, target):
        sx = random.uniform(self.min_scale, self.max_scale)
        sy = random.uniform(self.min_scale, self.max_scale)
        w, h = _get_size(img)
        new_w = max(1, int(round(w * sx)))
        new_h = max(1, int(round(h * sy)))

        img = F.resize(img, [new_h, new_w], antialias=self.antialias)

        # scale GT boxes if present
        if target is not None and "boxes" in target and target["boxes"] is not None:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= (new_w / max(w, 1))
            boxes[:, [1, 3]] *= (new_h / max(h, 1))
            target = {**target, "boxes": boxes}
        return img, target


@dataclass
class RandomResize:
    # Resize to a random short-side size; clamp longest side if needed
    short_side_choices: tuple[int, ...] = (320, 400, 480, 560, 640)
    max_long_side: int | None = 1024
    antialias: bool = True

    def __call__(self, img, target):
        short_side = random.choice(self.short_side_choices)
        w, h = _get_size(img)
        if min(w, h) == 0:
            return img, target

        scale = short_side / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # clamp if longest side exceeds max
        if self.max_long_side is not None and max(new_w, new_h) > self.max_long_side:
            clamp_scale = self.max_long_side / max(new_w, new_h)
            new_w = int(round(new_w * clamp_scale))
            new_h = int(round(new_h * clamp_scale))

        img = F.resize(img, [new_h, new_w], antialias=self.antialias)

        # scale GT boxes if present
        if target is not None and "boxes" in target and target["boxes"] is not None:
            sx = new_w / max(w, 1)
            sy = new_h / max(h, 1)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target = {**target, "boxes": boxes}

        return img, target
