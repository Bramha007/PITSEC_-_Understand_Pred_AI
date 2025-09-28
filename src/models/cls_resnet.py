# src/models/cls_resnet.py
from typing import Optional
import torch.nn as nn

try:
    from torchvision.models import (
        resnet18, resnet34, resnet50,
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
    )
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import resnet18, resnet34, resnet50  # type: ignore
    _HAS_WEIGHTS = False


def _load_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet18":
        if _HAS_WEIGHTS:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet18(weights=weights)
        return resnet18(pretrained=pretrained)
    if name == "resnet34":
        if _HAS_WEIGHTS:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet34(weights=weights)
        return resnet34(pretrained=pretrained)
    if name == "resnet50":
        if _HAS_WEIGHTS:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet50(weights=weights)
        return resnet50(pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {name}. Use resnet18|resnet34|resnet50.")


class Head2D(nn.Module):
    """Tiny head to regress two values; optional final activation."""
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


def build_resnet_backbone(model_name: str = "resnet18",
                          out_dim: int = 2,
                          pretrained: bool = True,
                          final_act: Optional[str] = None,
                          p_drop: float = 0.0) -> nn.Module:
    """
    Build a ResNet and replace the FC with a small regression head.
    If training on normalized targets in [0,1], set final_act='sigmoid'.
    """
    m = _load_backbone(model_name, pretrained=pretrained)
    in_features = m.fc.in_features  # type: ignore[attr-defined]
    m.fc = Head2D(in_features, out_dim=out_dim, final_act=final_act, p_drop=p_drop)  # type: ignore[attr-defined]
    return m
