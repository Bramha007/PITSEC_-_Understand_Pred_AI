from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from src.xai.core.roi_api import map_box_to_transformed, clip_box_to_image

def roi_cls_logit(model, img_t: torch.Tensor, pil_size: Tuple[int,int], box_xyxy: List[float]) -> float:
    device = next(model.parameters()).device
    W,H = pil_size
    images_list, _ = model.transform([img_t.to(device)], None)
    x = images_list.tensors
    Ht, Wt = images_list.image_sizes[0]
    feats = model.backbone(x)
    sx, sy = Wt/float(W), Ht/float(H)
    x1,y1,x2,y2 = box_xyxy
    box_tr = torch.tensor([[x1*sx, y1*sy, x2*sx, y2*sy]], dtype=torch.float32, device=device)
    pooled = model.roi_heads.box_roi_pool(feats, [box_tr], images_list.image_sizes)
    rep = model.roi_heads.box_head(pooled)
    logits, _ = model.roi_heads.box_predictor(rep)
    return float(logits[0,1].item())

def jitter_box(box, dx=0, dy=0):
    x1,y1,x2,y2 = box
    return [x1+dx, y1+dy, x2+dx, y2+dy]

def scale_box(box, scale=1.0):
    x1,y1,x2,y2 = box
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    w,h = (x2-x1)*scale, (y2-y1)*scale
    return [cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h]
