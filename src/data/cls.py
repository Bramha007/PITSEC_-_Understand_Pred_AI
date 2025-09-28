# src/data/cls.py
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

__all__ = ["collect_image_xml_pairs", "SquaresClassificationDatasetStream"]

def _ensure_wh_from_xml_or_image(rec: Dict[str, Any], xml_path: str, img_path: str) -> Dict[str, Any]:
    if "width" in rec and "height" in rec:
        return rec
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is not None:
            w = size.findtext("width"); h = size.findtext("height")
            if w and h:
                rec["width"] = int(float(w)); rec["height"] = int(float(h))
                return rec
    except Exception:
        pass
    with Image.open(img_path) as im:
        w, h = im.size
    rec["width"] = int(w); rec["height"] = int(h)
    return rec

def _parse_boxes_from_voc(xml_path: str) -> torch.Tensor:
    try:
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        if boxes:
            return torch.tensor(boxes, dtype=torch.float32)
    except Exception:
        pass
    return torch.zeros((0, 4), dtype=torch.float32)

def _imread(img_path: str) -> torch.Tensor:
    with Image.open(img_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        return F.to_tensor(im)

def collect_image_xml_pairs(img_dir: str, ann_dir: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Find (image, xml) pairs recursively.
    - Supports ONLY .bmp images (case-insensitive).
    - Requires an XML with the same stem somewhere under ann_dir (recursive).
    """
    img_root = Path(img_dir)
    ann_root = Path(ann_dir)
    if not img_root.exists():
        raise FileNotFoundError(f"Images folder not found: {img_root}")
    if not ann_root.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ann_root}")

    imgs: List[Path] = [p for p in img_root.rglob("*") if p.suffix.lower() == ".bmp"]

    # Build index of xmls by stem
    xml_index: Dict[str, List[Path]] = {}
    for xp in ann_root.rglob("*.xml"):
        xml_index.setdefault(xp.stem.lower(), []).append(xp)

    pairs: List[Tuple[str, str]] = []
    for ip in sorted(imgs):
        stem = ip.stem.lower()
        if stem in xml_index:
            xp = sorted(xml_index[stem], key=lambda p: len(str(p)))[0]  # prefer shortest path if multiple
            pairs.append((str(ip), str(xp)))

    if limit is not None:
        pairs = pairs[:limit]
    return pairs

class SquaresClassificationDatasetStream(Dataset):
    """
    Streaming classification/regression dataset from (img_path, xml_path) pairs.

    __getitem__ returns:
      img: FloatTensor [3,H,W] in [0,1]
      y  : FloatTensor [2] = (short_side_px, long_side_px) from the largest GT box.
          If no GT boxes exist, returns zeros([2]).
    """
    def __init__(self, pairs: List[Tuple[str, str]], transforms=None, canvas: bool = False, **kwargs):
        self.pairs = list(pairs)
        self.transforms = transforms
        self.canvas = canvas
        self._extra = kwargs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.pairs[idx]
        rec: Dict[str, Any] = {}
        rec = _ensure_wh_from_xml_or_image(rec, xml_path, img_path)

        img = _imread(img_path)

        boxes = _parse_boxes_from_voc(xml_path)
        if boxes.numel() > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            j = int(torch.argmax(areas).item())
            w = float(boxes[j, 2] - boxes[j, 0])
            h = float(boxes[j, 3] - boxes[j, 1])
            s, l = (w, h) if w <= h else (h, w)
            y = torch.tensor([s, l], dtype=torch.float32)
        else:
            y = torch.zeros(2, dtype=torch.float32)

        if self.transforms is not None:
            try:
                img, y = self.transforms(img, y)
            except TypeError:
                img = self.transforms(img)

        return img, y
