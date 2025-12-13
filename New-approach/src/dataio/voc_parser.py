# from pathlib import Path
# import xml.etree.ElementTree as ET

# def parse_voc(xml_path: str):
#     root = ET.parse(xml_path).getroot()
#     w = int(root.find("size/width").text)
#     h = int(root.find("size/height").text)
#     boxes = []
    
#     for obj in root.findall("object"):
#         bb = obj.find("bndbox")
#         xmin = max(0, int(float(bb.find("xmin").text)))
#         ymin = max(0, int(float(bb.find("ymin").text)))
#         xmax = int(float(bb.find("xmax").text))
#         ymax = int(float(bb.find("ymax").text))
        
#         if xmax > xmin and ymax > ymin:
#             boxes.append([xmin, ymin, xmax, ymax])

#     # Only return boxes, W, H, let the dataset calculate the label
#     return {"width": w, "height": h, "boxes": boxes}

# def paired_image_xml_list(img_dir, xml_dir, img_exts={".png", ".bmp", ".jpg", ".jpeg"}):
#     """Pairs image files with corresponding XML annotation files."""
#     img_dir, xml_dir = Path(img_dir), Path(xml_dir)
#     items = []
#     for p in sorted(img_dir.iterdir()):
#         if p.suffix.lower() in img_exts:
#             xml = xml_dir / (p.stem + ".xml")
#             if xml.exists():
#                 items.append((str(p), str(xml)))
#     return items


# File: src/dataio/voc_parser.py (CORRECTED for WxH extraction)

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple


# File: src/dataio/voc_parser.py (CORRECTED for WxH extraction)

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple


def parse_voc(xml_path: str) -> dict:
    """
    Parses a PASCAL VOC-style XML file, extracting canvas dimensions, boxes, 
    and the WxH string labels from the object attributes.
    """
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    # NEW: List to store WxH labels as strings (e.g., '32x64')
    labels_wxh_str = [] 
    
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        
        # --- Bounding Box Extraction ---
        xmin = max(0, int(float(bb.find("xmin").text)))
        ymin = max(0, int(float(bb.find("ymin").text)))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            
            # --- WxH Extraction from <attributes> ---
            attrs = obj.find("attributes")
            obj_w = attrs.find("width").text
            obj_h = attrs.find("height").text
            
            # Create a consistent WxH label string
            labels_wxh_str.append(f"{obj_w}x{obj_h}")

    # Return the new WxH labels along with boxes
    return {"width": w, "height": h, "boxes": boxes, "labels_wxh_str": labels_wxh_str} 


def paired_image_xml_list(img_dir, xml_dir, img_exts={".png", ".bmp", ".jpg", ".jpeg"}) -> List[Tuple[str, str]]:
    """Pairs image files with corresponding XML annotation files."""
    img_dir, xml_dir = Path(img_dir), Path(xml_dir)
    items = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in img_exts:
            xml = xml_dir / (p.stem + ".xml")
            if xml.exists():
                items.append((str(p), str(xml)))
    return items