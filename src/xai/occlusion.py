# src/xai/occlusion.py

import torch
import torch.nn.functional as F

def apply_occlusion(model, x: torch.Tensor, target=None, patch_size=16, stride=8):
    # Occlusion-Based Attribution
    model.eval()
    B, C, H, W = x.shape
    heatmap = torch.zeros((B, H, W), device=x.device)
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            x_occ = x.clone()
            x_occ[:, :, i:i+patch_size, j:j+patch_size] = 0
            out = model(x_occ)
            score = out.sum(dim=1) if target is None else (out*target).sum(dim=1)
            heatmap[:, i:i+patch_size, j:j+patch_size] += score.view(B,1,1)
    heatmap = heatmap / heatmap.max()
    return heatmap
