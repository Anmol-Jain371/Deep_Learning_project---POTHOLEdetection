"""
========================================================================================
🚗 RoadVision AI — Faster R-CNN Architecture & Factory
========================================================================================
Implements Faster R-CNN with ResNet-50-FPN backbone for two-class road damage detection:
  - Class 0: Background
  - Class 1: Pothole (D40)
Pretrained on COCO dataset, replacement classification/regression head.
========================================================================================
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights


def create_faster_rcnn_model(num_classes: int = 2, pretrained: bool = True):
    """
    Constructs Faster R-CNN with ResNet-50-FPN v2 backbone.
    Args:
        num_classes: Total classes including background (Default: 2)
        pretrained: Load ImageNet + COCO pre-trained weights
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace the pre-trained classification head with customized box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

