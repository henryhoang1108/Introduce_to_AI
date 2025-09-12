import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

def initialize_model(num_classes, device, id):
        if id == 18:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif id == 34:
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        for param in model.fc.parameters():
            param.requires_grad = True

        return model.to(device)