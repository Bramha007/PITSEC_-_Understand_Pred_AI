# src/models/cls_resnet.py

# ResNet Backbone With Custom Regression Head
# Supports resnet18|resnet34|resnet50 Backbones
# Head2D Allows Optional Dropout And Final Activation
# Logs Backbone Loading Status

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from src.utils.echo import echo_line

# Load ResNet Backbone From Local Weights Only
def _load_backbone(name: str, local_backbone_weights: str):
    # Normalize model name
    name = name.lower()

    # Require local weights
    if local_backbone_weights is None:
        raise ValueError("Must provide local_backbone_weights for offline use. No internet downloads allowed.")

    # Initialize model without pretrained weights
    if name == "resnet18":
        m = resnet18(weights=None)
    elif name == "resnet34":
        m = resnet34(weights=None)
    elif name == "resnet50":
        m = resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model_name: {name}. Use resnet18|resnet34|resnet50.")

    # Load local state dict
    local_loaded = False
    sd = torch.load(local_backbone_weights, map_location="cpu")
    try:
        m.load_state_dict(sd, strict=False)
        local_loaded = True
    except Exception as e:
        raise RuntimeError(f"Failed to load local backbone weights from {local_backbone_weights}") from e

    # Log loading status
    echo_line("CLS_MODEL", {
        "model_name": name,
        "local_backbone_loaded": local_loaded,
        "local_backbone_path": local_backbone_weights
    }, order=["model_name", "local_backbone_loaded"])

    return m

# Small Head To Regress Two Values With Optional Dropout And Final Activation
class Head2D(nn.Module):
    def __init__(self, in_features: int, out_dim: int = 2,
                 final_act: Optional[str] = None, p_drop: float = 0.0):
        super().__init__()
        layers = []
        if p_drop and p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        layers.append(nn.Linear(in_features, out_dim))
        if final_act is not None:
            act = final_act.lower()
            if act == "sigmoid":
                layers.append(nn.Sigmoid())
            elif act == "tanh":
                layers.append(nn.Tanh())
            elif act == "relu":
                layers.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Unknown final_act: {final_act}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Build ResNet Classifier With Regression Head
def build_resnet_classifier(model_name: str = "resnet18",
                            out_dim: int = 2,
                            final_act: Optional[str] = None,
                            p_drop: float = 0.0,
                            local_backbone_weights: str | None = None) -> nn.Module:
    # Load backbone (must provide local weights)
    m = _load_backbone(model_name, local_backbone_weights=local_backbone_weights)
    # Replace default fc with regression head
    in_features = m.fc.in_features  # type: ignore[attr-defined]
    m.fc = Head2D(in_features, out_dim=out_dim, final_act=final_act, p_drop=p_drop)  # type: ignore[attr-defined]
    return m
