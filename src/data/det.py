# src/data/det.py
# Detection Dataset For Squares Using VOC Annotations
# Provides Images, Bounding Boxes, And Labels For Training And Evaluation

from typing import List, Tuple, Dict, Any
from pathlib import Path
import PIL.Image as Image
import torch
from torch.utils.data import Dataset

from src.data.voc import parse_voc


# Dataset For Detection Using VOC Boxes
class SquaresDetectionDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], transforms=None):
        self.pairs = [(str(p[0]), str(p[1])) for p in pairs]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.pairs[idx]

        # Load Image
        img = Image.open(img_path).convert("RGB")

        # Parse VOC Annotations
        ann = parse_voc(xml_path)
        boxes = torch.tensor(ann.get("boxes", []), dtype=torch.float32)
        labels = torch.tensor(ann.get("labels", []), dtype=torch.int64)

        # Build Target Dictionary
        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "size": torch.tensor([img.size[1], img.size[0]], dtype=torch.int32),  # H, W
        }

        # Apply Optional Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# Collate Function For Detection Dataloaders
def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)
