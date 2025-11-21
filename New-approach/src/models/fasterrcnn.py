# import torchvision
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def build_fasterrcnn_cpu(num_classes=2,
#                          anchor_sizes=((4,), (8,), (16,), (32,), (64,)),
#                          anchor_ratios=((0.5,1.0,2.0),)*5):
#     # MobileNetV3 backbone runs well on CPU
#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
#     model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def build_fasterrcnn_cpu(num_classes=2):
#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model


# File: src/models/fasterrcnn.py (Renamed and Modularized)

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Tuple, List

def build_fasterrcnn(
    num_classes: int,
    backbone_weights: str = "DEFAULT",
    anchor_sizes: Tuple[Tuple[int, ...], ...] | None = None,
    anchor_ratios: Tuple[Tuple[float, ...], ...] | None = None
):
    """
    Builds a Faster R-CNN model with MobileNetV3 backbone.
    The model is device-agnostic; device selection happens in the training script.
    """
    # Load the pre-trained MobileNetV3 + FPN model
    # Weights are used for transfer learning unless set to None
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=backbone_weights if backbone_weights else None
    )

    # --- RPN Anchor Customization (Optional for small objects) ---
    if anchor_sizes and anchor_ratios:
        print("Note: Custom Anchor Generator applied.")
        model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
    
    # --- Classification Head Replacement ---
    # Get the number of input features for the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the existing head with a new one for our num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model