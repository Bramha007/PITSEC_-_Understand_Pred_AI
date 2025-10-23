# src/models/cls_resnet.py

# ResNet Backbone With Custom Two-Value Regression Head

from typing import Optional
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
)


# Load ResNet Backbone With Optional Pretrained Weights
def _load_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet18(weights=weights)
    if name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet34(weights=weights)
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet50(weights=weights)
    raise ValueError(f"Unsupported model_name: {name}. Use resnet18|resnet34|resnet50.")


# Small Head To Regress Two Values With Optional Final Activation
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


# Build ResNet And Replace FC Layer With Regression Head
def build_resnet_classifier(model_name: str = "resnet18",
                          out_dim: int = 2,
                          pretrained: bool = True,
                          final_act: Optional[str] = None,
                          p_drop: float = 0.0) -> nn.Module:
    m = _load_backbone(model_name, pretrained=pretrained)
    in_features = m.fc.in_features  # type: ignore[attr-defined]
    m.fc = Head2D(in_features, out_dim=out_dim, final_act=final_act, p_drop=p_drop)  # type: ignore[attr-defined]
    return m
