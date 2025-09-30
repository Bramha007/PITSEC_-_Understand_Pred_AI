# src/data/voc.py
# Utilities For Parsing VOC Annotations And Pairing Images With XML Files

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional


# Parse A VOC XML File Into Boxes And Labels
def parse_voc(xml_path: str) -> Dict[str, Any]:
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
            labels.append(1)  # Single-Class Dataset
    return {"boxes": boxes, "labels": labels}


# Return List Of (Image, XML) Pairs With Matching Stems
def paired_image_xml_list(
    img_dir: str | Path,
    ann_dir: str | Path,
    limit: Optional[int] = None
) -> List[Tuple[str, str]]:
    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ann_dir}")

    # Collect All BMP Images
    imgs: List[Path] = [p for p in img_dir.rglob("*") if p.suffix.lower() == ".bmp"]

    # Index XML Files By Stem
    xml_index: Dict[str, List[Path]] = {}
    for xp in ann_dir.rglob("*.xml"):
        xml_index.setdefault(xp.stem.lower(), []).append(xp)

    # Match Images To XMLs
    pairs: List[Tuple[str, str]] = []
    for ip in sorted(imgs):
        stem = ip.stem.lower()
        if stem in xml_index:
            xp = sorted(xml_index[stem], key=lambda p: len(str(p)))[0]
            pairs.append((str(ip), str(xp)))

    # Apply Limit If Provided
    if limit is not None:
        pairs = pairs[:limit]

    return pairs
