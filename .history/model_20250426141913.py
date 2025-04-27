from torch import nn
from models import VGG, ResNet18, ResNet50
from config import model_name

def get_model():
    if model_name == 'VGG19':
        return VGG('VGG19')
    elif model_name == 'Resnet18':
        return ResNet18()
    elif model_name == 'Resnet50':
        return ResNet50()
    else:
        raise ValueError(f"Unknown model: {model_name}")