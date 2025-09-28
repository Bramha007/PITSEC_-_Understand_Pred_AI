from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from src.xai.core.roi_api import map_box_to_transformed

def roi_gradcam_on_box(model, img_t: torch.Tensor, roi_box_xyxy: List[float]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    images_list, _ = model.transform([img_t.to(device)], None)
    x = images_list.tensors
    Ht, Wt = images_list.image_sizes[0]
    with torch.enable_grad():
        feats = model.backbone(x)  # OrderedDict level->Tensor
        boxes_tr = [torch.tensor([map_box_to_transformed(roi_box_xyxy, (img_t.shape[2], img_t.shape[1]), (Wt, Ht))], dtype=torch.float32, device=device)]
        pooled = model.roi_heads.box_roi_pool(feats, boxes_tr, images_list.image_sizes)
        pooled.retain_grad()
        rep = model.roi_heads.box_head(pooled)
        class_logits, _ = model.roi_heads.box_predictor(rep)
        scalar = class_logits[0,1]
        model.zero_grad(set_to_none=True)
        if pooled.grad is not None: pooled.grad.zero_()
        scalar.backward(retain_graph=False)
        Grad = pooled.grad[0]; Feat = pooled[0]
        w = Grad.mean(dim=(1,2), keepdim=True)
        cam = (w * Feat).sum(dim=0); cam = F.relu(cam)
    return {"pooled_cam": cam, "scalar": float(scalar.item()), "tr_size": (Ht, Wt)}
