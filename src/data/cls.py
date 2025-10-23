# src/data/cls.py

# Dataset For Classification/Regression Using VOC-Style Annotations
# Supports Optional Canvas Resizing, Letterboxing, And Custom Transforms
# Returns Image Tensor And Regression Targets (Short/Long Box Sides)

from __future__ import annotations

# Standard Library
from typing import List, Tuple, Optional

# Third-Party
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# Local Modules
from src.data.voc import parse_voc


# Classification/Regression Dataset Using VOC Annotations
class SquaresCLSData(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        canvas: int = 224,
        train: bool = False,
        use_padding_canvas: bool = True,
        transforms: Optional[object] = None,
    ):
        # Store Image-Annotation Pairs And Dataset Configuration
        self.pairs = [(str(p[0]), str(p[1])) for p in pairs]  # Normalize Paths To Strings
        self.canvas = int(canvas)                               # Canvas Size For Image Resize
        self.train = bool(train)                                # Training Mode Flag
        self.use_padding_canvas = bool(use_padding_canvas)      # Letterbox Flag
        self.transforms = transforms                            # Optional Transform Object

    def __len__(self) -> int:
        # Return Number Of Samples
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.pairs[idx]

        # Load Image And Convert To RGB
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Resize Image To Square Canvas (Letterbox Or Direct Resize)
        if self.use_padding_canvas:
            # Compute Scale And Resize Image Isotropically
            scale = min(self.canvas / w, self.canvas / h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_r = img.resize((new_w, new_h))

            # Create Black Canvas And Paste Resized Image Centered
            canvas = Image.new("RGB", (self.canvas, self.canvas), color=(0, 0, 0))
            off_x = (self.canvas - new_w) // 2
            off_y = (self.canvas - new_h) // 2
            canvas.paste(img_r, (off_x, off_y))
            img_t = F.to_tensor(canvas)
        else:
            # Direct Resize Without Letterboxing
            img_t = F.to_tensor(img.resize((self.canvas, self.canvas)))

        # Parse VOC Annotations
        ann = parse_voc(xml_path)
        if len(ann.get("boxes", [])) > 0:
            # Compute Short And Long Box Sides From First Ground Truth Box
            x1, y1, x2, y2 = ann["boxes"][0]
            short = float(min(x2 - x1, y2 - y1))
            long = float(max(x2 - x1, y2 - y1))
        else:
            # No Boxes Present, Default To Zero
            short, long = 0.0, 0.0

        # Construct Regression Target Tensor
        y = torch.tensor([short, long], dtype=torch.float32)

        # Apply Optional Transforms (May Modify Image And Target)
        if self.transforms is not None and hasattr(self.transforms, "__call__"):
            if self.train:
                # Training Mode Transform May Alter Both Image And Target
                img_t, y = self.transforms(img_t, y)
            else:
                # Evaluation Mode Transform Alters Only Image
                img_t = self.transforms(img_t)

        return img_t, y
