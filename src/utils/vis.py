# src/utils/vis.py

# Visualization Utilities For CLS Models
# Provides Heatmap Overlay Function Using Pillow And Matplotlib

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from typing import Any

def save_heatmap_overlay(img_tensor: Any, heatmap: Any, out_path: Path, alpha: float = 0.5):
    # Convert Image Tensor To HxWxC PIL Image
    if hasattr(img_tensor, "cpu"):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor)

    if img.ndim == 3 and img.shape[0] in {1, 3}:
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img)

    # Convert Heatmap To HxW 2D Array
    if hasattr(heatmap, "cpu"):
        hm = heatmap.detach().cpu().numpy()
    else:
        hm = np.array(heatmap)

    # Squeeze all singleton dimensions
    hm = np.squeeze(hm)

    # Ensure heatmap is 2D
    if hm.ndim != 2:
        hm = hm[:, :, 0] if hm.ndim == 3 else hm.reshape(hm.shape[-2], hm.shape[-1])

    # Clip and convert to float32
    hm = hm.astype(np.float32)
    hm = np.clip(hm, 0, 1)

    # Apply Colormap
    colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(hm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Resize heatmap if needed to match image
    if heatmap_pil.size != img_pil.size:
        heatmap_pil = heatmap_pil.resize(img_pil.size, resample=Image.BILINEAR)

    # Overlay Heatmap On Image
    overlay = Image.blend(img_pil, heatmap_pil, alpha=alpha)
    overlay.save(out_path)
