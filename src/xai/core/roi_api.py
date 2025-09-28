from typing import Tuple, List
import torch

def map_box_to_transformed(box_xyxy: List[float], src_wh: Tuple[int,int], dst_wh: Tuple[int,int]):
    """Scale xyxy from original (W,H) to transformed (Wt,Ht)."""
    W,H = src_wh; Wt,Ht = dst_wh
    sx, sy = float(Wt)/float(W), float(Ht)/float(H)
    x1,y1,x2,y2 = box_xyxy
    return [x1*sx, y1*sy, x2*sx, y2*sy]

def clip_box_to_image(box, W, H):
    x1,y1,x2,y2 = box
    x1 = float(max(0, min(W-1, x1))); y1 = float(max(0, min(H-1, y1)))
    x2 = float(max(0, min(W,   x2))); y2 = float(max(0, min(H,   y2)))
    if x2 <= x1 + 1: x2 = min(W, x1 + 1)
    if y2 <= y1 + 1: y2 = min(H, y1 + 1)
    return [x1,y1,x2,y2]
