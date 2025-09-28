# src/data/voc.py
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional

def parse_voc(xml_path: str) -> Dict[str, Any]:
    """
    Minimal VOC XML parser -> dict with 'boxes' and 'labels'.
    Boxes are xyxy in pixel coords; single class -> label=1.
    """
    root = ET.parse(xml_path).getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
    return {"boxes": boxes, "labels": labels}

def paired_image_xml_list(img_dir: str | Path,
                          ann_dir: str | Path,
                          limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Return list of (bmp_image_path, xml_path) with SAME stem.
    - ONLY .bmp images (case-insensitive) are considered.
    - XML can live anywhere under ann_dir; shortest path wins if multiple.
    """
    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ann_dir}")

    imgs: List[Path] = [p for p in img_dir.rglob("*") if p.suffix.lower() == ".bmp"]

    xml_index: Dict[str, List[Path]] = {}
    for xp in ann_dir.rglob("*.xml"):
        xml_index.setdefault(xp.stem.lower(), []).append(xp)

    pairs: List[Tuple[str, str]] = []
    for ip in sorted(imgs):
        stem = ip.stem.lower()
        if stem in xml_index:
            xp = sorted(xml_index[stem], key=lambda p: len(str(p)))[0]
            pairs.append((str(ip), str(xp)))

    if limit is not None:
        pairs = pairs[:limit]
    return pairs
