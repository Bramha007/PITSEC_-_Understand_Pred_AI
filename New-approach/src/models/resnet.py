# import torchvision
# import torch.nn as nn

# def build_resnet_classifier(num_classes=5):
#     model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
#     in_features = model.fc.in_features
#     model.fc = nn.Linear(in_features, num_classes)
#     return model

# For classification of square sizes with modular class count


import torchvision
import torch.nn as nn
from src.setup.config_cls import NUM_CLS_CLASSES # Import the modular class count

def build_resnet_classifier(num_classes: int = NUM_CLS_CLASSES):
    """
    Builds a ResNet-18 model customized for size classification.
    """
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    # Use the dynamically supplied num_classes
    model.fc = nn.Linear(in_features, num_classes)
    return model
