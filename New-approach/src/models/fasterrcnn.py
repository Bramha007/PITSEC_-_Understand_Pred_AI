import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Tuple, List

def build_fasterrcnn(
    num_classes: int,
    backbone_weights: str = "DEFAULT",
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


    # --- Classification Head Replacement ---
    # Get the number of input features for the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the existing head with a new one for our num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
