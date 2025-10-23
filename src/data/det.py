# src/data/det.py

# Dataset For Object Detection Using VOC-Style Bounding Boxes
# Supports Optional Image/Target Transforms And PyTorch Dataloader Collation
# Returns Image And Target Dictionary With Boxes, Labels, And Metadata

from __future__ import annotations

# Standard Library
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Third-Party
import PIL.Image as Image
import torch
from torch.utils.data import Dataset

# Local Modules
from src.data.voc import parse_voc


# Detection Dataset Using VOC Annotations
class SquaresDetectionDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], transforms=None):
        # Store Image-Annotation Pairs And Optional Transform
        self.pairs = [(str(p[0]), str(p[1])) for p in pairs]  # Normalize Paths To Strings
        self.transforms = transforms                            # Optional Transform Object

    def __len__(self) -> int:
        # Return Number Of Samples
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.pairs[idx]

        # Load Image And Convert To RGB
        img = Image.open(img_path).convert("RGB")

        # Parse VOC Annotations For Bounding Boxes And Labels
        ann = parse_voc(xml_path)
        boxes = torch.tensor(ann.get("boxes", []), dtype=torch.float32)  # Nx4 Float Tensor
        labels = torch.tensor(ann.get("labels", []), dtype=torch.int64)  # Nx1 Int Tensor

        # Build Target Dictionary Required By Detection Models
        target: Dict[str, Any] = {
            "boxes": boxes,                                          # Ground Truth Boxes
            "labels": labels,                                        # Class Labels
            "image_id": torch.tensor([idx]),                         # Image Index
            "size": torch.tensor([img.size[1], img.size[0]], dtype=torch.int32),  # H, W
        }

        # Apply Optional Transform (May Modify Image And Target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# Collate Function For Detection Dataloaders
def collate_fn(batch):
    # Unzip Images And Targets And Convert To Lists
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)
