# src/data/det.py
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.voc import parse_voc


def collate_fn(batch):
    """
    Picklable collate function (required on Windows spawn).
    Groups a list of (img, target) into tuple(imgs, targets).
    """
    return tuple(zip(*batch))


class SquaresDetectionDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], transforms=None):
        self.pairs = list(pairs)
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.pairs[idx]

        # Keep PIL in the dataset; let transforms.ToTensor() handle tensor conversion
        with Image.open(img_path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            img = im.copy()  # detach from context manager

        rec = parse_voc(xml_path)

        # Use as_tensor to avoid the “copy construct from a tensor” warning
        boxes = torch.as_tensor(rec["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(rec["labels"], dtype=torch.int64)
        target: Dict[str, Any] = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
