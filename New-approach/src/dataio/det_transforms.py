import random
from torchvision.transforms import functional as F

class ToTensor:
    def __call__(self, img, target):
        """Converts PIL Image to normalized float32 tensor."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = F.to_tensor(img)  # float32 [0,1]
        return img, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, target):
        """Randomly flips the image and updates bounding boxes."""
        if random.random() < self.p:
            w, _ = F.get_image_size(img)
            
            # Boxes should always be CPU tensors at this stage
            boxes = target["boxes"].clone()
            
            # Flip coordinates: x' = W - x
            # [x_min, y_min, x_max, y_max] -> [W-x_max, y_min, W-x_min, y_max]
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            
            target["boxes"] = boxes
            img = F.hflip(img)
        return img, target

class ClampBoxes:
    def __call__(self, img, target):
        """Clamps box coordinates to be within image bounds (0 to W/H)."""
        w, h = F.get_image_size(img)
        boxes = target["boxes"]
        
        # Clamp x-coordinates between 0 and w
        boxes[:, 0].clamp_(min=0, max=w)
        boxes[:, 2].clamp_(min=0, max=w)
        
        # Clamp y-coordinates between 0 and h
        boxes[:, 1].clamp_(min=0, max=h)
        boxes[:, 3].clamp_(min=0, max=h)

        target["boxes"] = boxes
        return img, target


class Compose:
    def __init__(self, ts): self.ts = ts
    def __call__(self, img, target):
        """Applies a sequence of transforms."""
        for t in self.ts: img, target = t(img, target)
        return img, target