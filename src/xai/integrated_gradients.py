# src/xai/integrated_gradients.py

# Integrated Gradients Attribution For CLS Models (Regression)

from captum.attr import IntegratedGradients
import torch

def apply_ig(model, x: torch.Tensor, target=None):
    # Class-Agnostic Integrated Gradients
    model.eval()
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, n_steps=50)
    attr = attr.mean(dim=1, keepdim=True)
    attr = torch.relu(attr)
    attr = attr / (attr.max() + 1e-8)
    return attr
