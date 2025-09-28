# src/models/det_fasterrcnn.py
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

# Local backbone checkpoint path (next to this file)
_BACKBONE_PATH = Path(__file__).resolve().parent / "det_backbone.pth"


def _load_local_backbone_or_skip(model):
    if _BACKBONE_PATH.exists():
        try:
            state = torch.load(_BACKBONE_PATH, map_location="cpu")
            missing, unexpected = model.backbone.body.load_state_dict(state, strict=False)
            print(f"[det_fasterrcnn] Loaded backbone: {_BACKBONE_PATH.name} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            print(f"[det_fasterrcnn] WARNING: failed to load local backbone: {e}. Using random init.")
    else:
        print(f"[det_fasterrcnn] Local backbone not found at {_BACKBONE_PATH}. Using random init.")


def _normalize_anchor_sizes(anchor_sizes):
    """
    Ensure each FPN level has the SAME number of sizes (required by torchvision RPN head).
    We use the length of level 0 as reference: pad by repeating the last size, or truncate.
    """
    if not isinstance(anchor_sizes, (list, tuple)) or not anchor_sizes:
        return anchor_sizes
    ref_len = len(anchor_sizes[0])
    norm = []
    changed = False
    for lvl, sizes in enumerate(anchor_sizes):
        s = list(sizes)
        if len(s) < ref_len:
            s = s + [s[-1]] * (ref_len - len(s))  # pad by repeating last
            changed = True
        elif len(s) > ref_len:
            s = s[:ref_len]                        # truncate
            changed = True
        norm.append(tuple(s))
    if changed:
        print(f"[det_fasterrcnn] Normalized anchor sizes per level to length {ref_len}: {norm}")
    return norm


def build_fasterrcnn(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )
    _load_local_backbone_or_skip(model)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def build_fasterrcnn_custom(cfg: dict):
    num_classes = cfg.get("num_classes", 2)
    anchor_sizes = cfg.get("anchor_sizes", [[32, 64, 128, 256, 512]])
    anchor_ratios = cfg.get("anchor_ratios", [0.5, 1.0, 2.0])

    # Make sizes consistent across FPN levels
    anchor_sizes = _normalize_anchor_sizes(anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=anchor_ratios
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=cfg.get("box_detections_per_img", 100),
        rpn_pre_nms_top_n_train=cfg.get("rpn_pre_nms_topn_train", 2000),
        rpn_pre_nms_top_n_test=cfg.get("rpn_pre_nms_topn_test", 1000),
        rpn_post_nms_top_n_train=cfg.get("rpn_post_nms_top_n_train", 2000),
        rpn_post_nms_top_n_test=cfg.get("rpn_post_nms_top_n_test", 1000),
    )
    _load_local_backbone_or_skip(model)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model
