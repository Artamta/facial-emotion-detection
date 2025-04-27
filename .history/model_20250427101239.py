from models.vgg import VGG
from models.resnet import ResNet18

class MyCustomModel:
    @staticmethod
    def get_model(model_name, num_classes=7):
        if model_name == 'VGG19':
            return VGG('VGG19')
        elif model_name == 'ResNet18':
            return ResNet18()
        else:
            raise ValueError(f"Unsupported model: {model_name}")