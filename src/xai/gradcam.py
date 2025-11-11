# src/xai/gradcam.py

import torch
import torch.nn.functional as F

def apply_gradcam(model, x: torch.Tensor, target_layer=None):
    # Returns Grad-CAM heatmaps for input batch x.
    # model: classifier model
    # x: [B, C, H, W]
    # target_layer: optional, if None use last conv layer
    model.eval()
    heatmaps = []

    if target_layer is None:
        # Default To Last Conv Layer
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
    if target_layer is None:
        raise ValueError("No conv layer found for Grad-CAM.")

    def forward_hook(module, input, output):
        forward_hook.features = output
    forward_hook.features = None
    target_layer.register_forward_hook(forward_hook)

    out = model(x)
    if out.shape[1] == 1:
        scores = out
    else:
        scores = out.max(1)[0]  # Take Max Logit

    model.zero_grad()
    scores.sum().backward(retain_graph=True)
    gradients = target_layer.weight.grad if hasattr(target_layer, "weight") else None
    if gradients is None:
        gradients = next(model.parameters()).grad
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    fmap = forward_hook.features.detach()
    for i in range(fmap.shape[0]):
        heatmap = torch.sum(pooled_grads.view(1, -1, 1, 1) * fmap[i], dim=0)
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmaps.append(heatmap.unsqueeze(0))
    return torch.stack(heatmaps)
