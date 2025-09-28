import numpy as np
import torch
import matplotlib.pyplot as plt

from src.constants.norms import IMAGENET_MEAN_T, IMAGENET_STD_T

COLORMAP = plt.cm.jet

@torch.no_grad()
def denorm(chw: torch.Tensor) -> torch.Tensor:
    # Accept [3,H,W] or [B,3,H,W]; return same shape clamped to [0,1]
    if chw.dim() == 3:
        x = chw.unsqueeze(0)
    else:
        x = chw
    x = (x * IMAGENET_STD_T.to(x.device)) + IMAGENET_MEAN_T.to(x.device)
    x = x.clamp(0,1)
    return x if chw.dim()==4 else x[0]

def overlay_cam(img_chw: torch.Tensor, hm_hw: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """img_chw: [3,H,W] float(0..1) (normalized ok — will be denormed inside)
       hm_hw: [H,W] float(0..1)
       returns HxWx3 float(0..1) numpy
    """
    img = denorm(img_chw).permute(1,2,0).cpu().numpy()
    hm  = hm_hw.detach().cpu().numpy()
    cmap = COLORMAP(hm)[..., :3]
    out = (1-alpha)*img + alpha*cmap
    return np.clip(out, 0, 1)

def draw_box(ax, box, color="white", lw=1.5, ls="-"):
    x1,y1,x2,y2 = box
    import matplotlib.patches as patches
    ax.add_patch(patches.Rectangle((x1,y1),(x2-x1),(y2-y1), fill=False, edgecolor=color, lw=lw, linestyle=ls))
