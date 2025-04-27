from models import VGG, ResNet18

def get_model(model_name='VGG19'):
    if model_name == 'VGG19':
        return VGG('VGG19')
    elif model_name == 'ResNet18':
        return ResNet18()
    else:
        raise ValueError(f"Unknown model: {model_name}")