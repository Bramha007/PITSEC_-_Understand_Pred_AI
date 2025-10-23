# src/data/voc.py

# Utilities For Parsing VOC-Style XML Annotations
# Supports Single-Class Detection Dataset With Boxes And Labels
# Provides Helper To Pair Images With Corresponding XML Files Deterministically

# Standard Library
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional


# Parse A VOC XML File Into Boxes And Labels
def parse_voc(xml_path: str) -> Dict[str, Any]:
    # Load XML And Get Root Element
    root = ET.parse(xml_path).getroot()
    boxes = []
    labels = []

    # Iterate Over Object Elements
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))

        # Only Keep Valid Boxes
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
    # Convert To Path Objects
    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)

    # Validate Existence
    if not img_dir.exists():
        raise FileNotFoundError(f"Images Folder Not Found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations Folder Not Found: {ann_dir}")

    # Collect All BMP Images Recursively
    imgs: List[Path] = [p for p in img_dir.rglob("*") if p.suffix.lower() == ".bmp"]

    # Build Index Of XML Files By Stem
    xml_index: Dict[str, List[Path]] = {}
    for xp in ann_dir.rglob("*.xml"):
        xml_index.setdefault(xp.stem.lower(), []).append(xp)

    # Match Images To XML Files Deterministically
    pairs: List[Tuple[str, str]] = []
    for ip in sorted(imgs):
        stem = ip.stem.lower()
        if stem in xml_index:
            # Pick XML With Shortest Path If Multiple Exist
            xp = sorted(xml_index[stem], key=lambda p: len(str(p)))[0]
            pairs.append((str(ip), str(xp)))

    # Apply Optional Limit
    if limit is not None:
        pairs = pairs[:limit]

    return pairs
