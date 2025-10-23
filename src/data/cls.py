# src/data/cls.py

# Classification/Regression Dataset From VOC Boxes With Optional Transforms

from typing import List, Tuple, Optional
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from src.data.voc import parse_voc


# Dataset For Classification/Regression Using VOC Annotations
class SquaresCLSData(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        canvas: int = 224,
        train: bool = False,
        use_padding_canvas: bool = True,
        transforms: Optional[object] = None,
    ):
        self.pairs = [(str(p[0]), str(p[1])) for p in pairs]
        self.canvas = int(canvas)
        self.train = bool(train)
        self.use_padding_canvas = bool(use_padding_canvas)
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Resize To Square Canvas (Letterbox) Or Isotropic Resize
        if self.use_padding_canvas:
            scale = min(self.canvas / w, self.canvas / h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_r = img.resize((new_w, new_h))
            canvas = Image.new("RGB", (self.canvas, self.canvas), color=(0, 0, 0))
            off_x = (self.canvas - new_w) // 2
            off_y = (self.canvas - new_h) // 2
            canvas.paste(img_r, (off_x, off_y))
            img_t = F.to_tensor(canvas)
        else:
            img_t = F.to_tensor(img.resize((self.canvas, self.canvas)))

        # Extract Target From First GT Box
        ann = parse_voc(xml_path)
        if len(ann.get("boxes", [])) > 0:
            x1, y1, x2, y2 = ann["boxes"][0]
            short = float(min(x2 - x1, y2 - y1))
            long = float(max(x2 - x1, y2 - y1))
        else:
            short, long = 0.0, 0.0
        y = torch.tensor([short, long], dtype=torch.float32)

        # Apply Optional Transforms (CLSTransform handles img or img+label)
        if self.transforms is not None:
            if hasattr(self.transforms, "__call__"):
                if self.train:
                    img_t, y = self.transforms(img_t, y)
                else:
                    img_t = self.transforms(img_t)

        return img_t, y
