# src/transforms/det.py

# Main detection transforms used in training/eval.
# Minimal + stable; experimental ones live in det_experimental.py.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any
import random
import torch
from torchvision.transforms import functional as F

__all__ = [
    "Compose",
    "ToTensor",
    "RandomHorizontalFlip",
    "ResizeKeepRatio",
    "ClampBoxes",
]


def _get_size(img) -> Tuple[int, int]:
    # Return (W,H) for PIL or Tensor images
    w, h = F.get_image_size(img)
    return int(w), int(h)


class Compose:
    # Chain multiple transforms
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, img, target: Dict[str, Any]):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    # Convert PIL -> FloatTensor [0,1], keep tensor as-is
    def __call__(self, img, target):
        if isinstance(img, torch.Tensor):
            return img, target
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        return F.to_tensor(img), target


@dataclass
class RandomHorizontalFlip:
    # Flip with probability p
    p: float = 0.5

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target

        w, _ = _get_size(img)
        img = F.hflip(img)

        # flip x-coordinates of boxes
        if target is not None and "boxes" in target and target["boxes"] is not None:
            boxes = target["boxes"]
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
            target = {**target, "boxes": boxes}
        return img, target


@dataclass
class ResizeKeepRatio:
    # Resize to short_side, keep ratio; clamp by max_long_side if given
    short_side: int
    max_long_side: int | None = None
    antialias: bool = True

    def __call__(self, img, target):
        w, h = _get_size(img)
        if min(w, h) == 0:
            return img, target

        scale = self.short_side / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        if self.max_long_side is not None and max(new_w, new_h) > self.max_long_side:
            clamp_scale = self.max_long_side / max(new_w, new_h)
            new_w = int(round(new_w * clamp_scale))
            new_h = int(round(new_h * clamp_scale))

        if new_w == w and new_h == h:
            return img, target

        img = F.resize(img, [new_h, new_w], antialias=self.antialias)

        # scale GT boxes
        if target is not None and "boxes" in target and target["boxes"] is not None:
            sx = new_w / max(w, 1)
            sy = new_h / max(h, 1)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target = {**target, "boxes": boxes}
        return img, target


@dataclass
class ClampBoxes:
    # Clamp boxes inside image; enforce min size
    min_size: float = 1.0

    def __call__(self, img, target):
        if target is None or "boxes" not in target or target["boxes"] is None:
            return img, target

        w, h = _get_size(img)
        boxes = target["boxes"].clone()

        # clamp coords
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0.0, max=float(w))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0.0, max=float(h))

        # ensure nonzero width/height
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x2 = torch.maximum(x2, x1 + self.min_size)
        y2 = torch.maximum(y2, y1 + self.min_size)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        target = {**target, "boxes": boxes}
        return img, target
