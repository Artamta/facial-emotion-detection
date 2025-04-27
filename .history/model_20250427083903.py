import torch.nn as nn
from models import VGG  # Assuming you already have the VGG model defined

class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        self.model = VGG('VGG19')

    def forward(self, x):
        return self.model(x)