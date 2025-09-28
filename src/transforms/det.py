# src/transforms/det.py
import random
import torch
from torchvision.transforms import functional as F


class ToTensor:
    def __call__(self, img, target):
        # If already a tensor, just return it
        if isinstance(img, torch.Tensor):
            return img, target
        # Otherwise, PIL → RGB → tensor
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        img = F.to_tensor(img)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            # F.get_image_size works for PIL or Tensor
            w, _ = F.get_image_size(img)
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            img = F.hflip(img)
        return img, target


class AnisotropicScale:
    """
    Randomly scale width and height independently.
    Accepts scale_range=(min,max) or min_scale/max_scale.
    """
    def __init__(self, scale_range=None, min_scale=None, max_scale=None):
        if scale_range is not None:
            self.min_scale, self.max_scale = scale_range
        else:
            self.min_scale = 0.7 if min_scale is None else min_scale
            self.max_scale = 1.3 if max_scale is None else max_scale

    def __call__(self, img, target):
        sx = random.uniform(self.min_scale, self.max_scale)
        sy = random.uniform(self.min_scale, self.max_scale)
        w, h = F.get_image_size(img)
        new_w = max(1, int(round(w * sx)))
        new_h = max(1, int(round(h * sy)))
        img = F.resize(img, [new_h, new_w], antialias=True)
        if "boxes" in target and target["boxes"] is not None:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target["boxes"] = boxes
        return img, target


class RandomResize:
    """Pick target short side from `sizes` and resize isotropically."""
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        w, h = F.get_image_size(img)
        scale = size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = F.resize(img, (new_h, new_w), antialias=True)
        if "boxes" in target:
            boxes = target["boxes"] * torch.tensor([scale, scale, scale, scale])
            target["boxes"] = boxes
        return img, target


class ClampBoxes:
    def __call__(self, img, target):
        if "boxes" not in target or target["boxes"] is None:
            return img, target
        boxes = target["boxes"].clone().to(dtype=torch.float32)
        w, h = F.get_image_size(img)
        boxes[:, 0].clamp_(0, max(w - 1, 0))
        boxes[:, 2].clamp_(0, max(w - 1, 0))
        boxes[:, 1].clamp_(0, max(h - 1, 0))
        boxes[:, 3].clamp_(0, max(h - 1, 0))
        fix_x = boxes[:, 2] <= boxes[:, 0]
        boxes[fix_x, 2] = torch.clamp(boxes[fix_x, 0] + 1.0, max=w - 1)
        fix_y = boxes[:, 3] <= boxes[:, 1]
        boxes[fix_y, 3] = torch.clamp(boxes[fix_y, 1] + 1.0, max=h - 1)
        target["boxes"] = boxes
        return img, target


class Compose:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, img, target):
        for t in self.ts:
            img, target = t(img, target)
        return img, target
