import torch.nn as nn
from models import VGG

class MyCustomModel(nn.Module):
    def __init__(self, model_name='VGG19'):
        super(MyCustomModel, self).__init__()
        if model_name == 'VGG19':
            self.model = VGG('VGG19')
        else:
            raise ValueError("Unsupported model: %s" % model_name)

    def forward(self, x):
        return self.model(x)