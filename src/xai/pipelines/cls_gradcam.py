from typing import Dict, Any
import torch
import numpy as np
from src.xai.core.gradcam import GradCAM

def run_cls_gradcam_pipeline(model, x: torch.Tensor, target_class: int | None = None) -> Dict[str, Any]:
    cammer = GradCAM(model, model.layer4[-1])
    cam, pred_cls, probs = cammer(x, target_class=target_class)
    cammer.remove()
    return {"cam": cam[0,0], "pred": int(pred_cls), "probs": probs[0].cpu().numpy()}
