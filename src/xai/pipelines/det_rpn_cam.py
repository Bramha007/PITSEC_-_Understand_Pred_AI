from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

def rpn_cell_gradcam(model, img_t: torch.Tensor, target_xyxy: List[float]) -> Dict[str, Any]:
    device = next(model.parameters()).device
    images_list, _ = model.transform([img_t.to(device)], None)
    x = images_list.tensors
    Ht, Wt = images_list.image_sizes[0]
    with torch.enable_grad():
        feat_od = model.backbone(x)
        feat_list = list(feat_od.values())
        for t in feat_list: t.retain_grad()
        objectness_list, _ = model.rpn.head(feat_list)

        x1,y1,x2,y2 = target_xyxy
        xc, yc = 0.5*(x1+x2), 0.5*(y1+y2)

        best = None
        for l_idx, feat in enumerate(feat_list):
            _, C, Hl, Wl = feat.shape
            stride_y = float(Ht)/Hl; stride_x = float(Wt)/Wl
            j = int(np.clip(np.floor(xc/stride_x), 0, Wl-1))
            i = int(np.clip(np.floor(yc/stride_y), 0, Hl-1))
            logits = objectness_list[l_idx][0]  # [A,Hl,Wl]
            cell_logits = logits[:, i, j]       # [A]
            a_idx = int(torch.argmax(cell_logits).item())
            s = float(cell_logits[a_idx].item())
            if (best is None) or (s > best[0]): best = (s, l_idx, a_idx, i, j)

        if best is None:
            return {"ok": False, "msg": "no rpn anchor selected"}

        s, l_idx, a_idx, i, j = best
        scalar = objectness_list[l_idx][0, a_idx, i, j]
        model.zero_grad(set_to_none=True)
        for t in feat_list:
            if t.grad is not None: t.grad.zero_()
        scalar.backward(retain_graph=False)

        Feat = feat_list[l_idx][0]
        Grad = feat_list[l_idx].grad[0]
        if Grad is None:
            return {"ok": False, "msg": "no grads on feature level"}

        w = Grad.mean(dim=(1,2), keepdim=True)
        cam = (w * Feat).sum(dim=0); cam = F.relu(cam)

    return {"ok": True, "level_idx": int(l_idx), "obj_logit": float(s),
            "cam_level": cam, "tr_size": (Ht, Wt)}
