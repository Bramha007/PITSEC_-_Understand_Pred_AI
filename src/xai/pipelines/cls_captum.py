from typing import Dict, Any
import torch
from src.xai.core.captum_wrappers import captum_ig, captum_saliency

def run_cls_captum_pipeline(model, x: torch.Tensor, target: int, steps: int = 64) -> Dict[str, Any]:
    x = x.clone().requires_grad_(True)
    atts_ig = captum_ig(model, x, target=target, steps=steps)
    atts_ig = atts_ig.squeeze(0).abs().sum(0)   # [H,W]
    atts_ig = (atts_ig - atts_ig.min()) / (atts_ig.max() - atts_ig.min() + 1e-6)

    x.grad = None; x.requires_grad_(True)
    atts_sal = captum_saliency(model, x, target=target)
    atts_sal = atts_sal.squeeze(0).abs().sum(0)
    atts_sal = (atts_sal - atts_sal.min()) / (atts_sal.max() - atts_sal.min() + 1e-6)

    return {"ig": atts_ig, "saliency": atts_sal}
