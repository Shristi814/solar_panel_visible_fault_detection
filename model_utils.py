# model_utils.py
import torch.nn as nn
from torchvision import models

IMG_SIZE = 224


def build_model(name, num_classes, pretrained=True):
    if name == "mobilenet_v3":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=w)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(name)
    return model