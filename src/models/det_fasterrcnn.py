# src/models/det_fasterrcnn.py

# Faster R-CNN Builder (Torch 2.6 / Torchvision 0.21) With Custom Anchors And Optional Local Backbone Weights

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from src.utils.echo import echo_line


def _normalize_anchor_spec(
    sizes: Optional[Sequence],
    aspect_ratios: Optional[Sequence],
    num_levels: int = 5,
) -> Tuple[Optional[Tuple[Tuple[int, ...], ...]], Optional[Tuple[Tuple[float, ...], ...]]]:
    # Normalize Sizes Into Tuple-Of-Tuples Per FPN Level
    def _sizes_to_tuple_tuple(x):
        if x is None: return None
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return tuple(tuple(int(max(1, int(v))) for v in lvl) for lvl in x)
        base = tuple(int(max(1, int(v))) for v in x)
        return tuple([base] * num_levels)

    # Normalize Ratios Into Tuple-Of-Tuples Per FPN Level
    def _ratios_to_tuple_tuple(x):
        if x is None: return None
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return tuple(tuple(float(v) for v in lvl) for lvl in x)
        base = tuple(float(v) for v in x)
        return tuple([base] * num_levels)

    sizes_tt = _sizes_to_tuple_tuple(sizes)
    ratios_tt = _ratios_to_tuple_tuple(aspect_ratios)

    # Require Both Or Neither
    if (sizes_tt is None) ^ (ratios_tt is None):
        raise ValueError("Provide Both 'anchor_sizes' And 'anchor_aspect_ratios' Or Neither.")

    # Validate Per-Level Lengths
    if sizes_tt and ratios_tt and len(sizes_tt) != len(ratios_tt):
        raise ValueError(f"Anchor Levels Mismatch: sizes={len(sizes_tt)} vs ratios={len(ratios_tt)}.")

    # Validate Positive Sizes
    if sizes_tt:
        for i, s in enumerate(sizes_tt):
            if any(v <= 0 for v in s):
                raise ValueError(f"Anchor Sizes Must Be Positive (Level {i}: {s}).")

    return sizes_tt, ratios_tt


def _format_anchor_summary(
    sizes_tt: Optional[Tuple[Tuple[int, ...], ...]],
    ratios_tt: Optional[Tuple[Tuple[float, ...], ...]],
) -> list[Dict[str, Any]]:
    # Create A Compact Summary For Logging
    if sizes_tt is None or ratios_tt is None: return []
    return [{"level": i, "sizes": list(map(int, s)), "ratios": list(map(float, r))}
            for i, (s, r) in enumerate(zip(sizes_tt, ratios_tt))]


def build_fasterrcnn(
    num_classes: int = 2,
    anchor_sizes: Optional[Sequence] = None,
    anchor_aspect_ratios: Optional[Sequence] = None,
    weights: Optional[str] = None,                      # Keep None/'none' To Avoid Downloads
    trainable_backbone_layers: Optional[int] = None,
    local_backbone_weights: Optional[str] = None,       # Path To weights/resnet50-0676ba61.pth
) -> torch.nn.Module:
    # Anchor Generator (Optional Custom Anchors)
    sizes_tt, ratios_tt = _normalize_anchor_spec(anchor_sizes, anchor_aspect_ratios, num_levels=5)
    rpn_anchor_generator = None
    if sizes_tt and ratios_tt:
        rpn_anchor_generator = AnchorGenerator(sizes=sizes_tt, aspect_ratios=ratios_tt)

    # Build Faster R-CNN With Optional Custom Anchors
    model = fasterrcnn_resnet50_fpn(
        weights=None if weights in (None, "", "none") else weights,
        weights_backbone=None,                           # No Auto-Download
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Optionally Load Local Backbone Weights
    local_loaded = False
    if local_backbone_weights:
        sd = torch.load(local_backbone_weights, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.backbone.body.load_state_dict(sd, strict=False)
            local_loaded = True
        except Exception:
            model.backbone.load_state_dict(sd, strict=False)
            local_loaded = True

    # Log Model Summary
    echo_line("DET_MODEL", {
        "num_classes": int(num_classes),
        "custom_anchors": bool(rpn_anchor_generator is not None),
        "anchors": _format_anchor_summary(sizes_tt, ratios_tt),
        "local_backbone_loaded": bool(local_loaded),
        "local_backbone_path": local_backbone_weights or "",
    }, order=["num_classes","custom_anchors","local_backbone_loaded"])

    return model
