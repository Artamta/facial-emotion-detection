import torch
import torch.nn as nn
import torch.nn.functional as F
from config import resize_x, resize_y, input_channels

class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super(MyCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size dynamically
        dummy_input = torch.randn(1, input_channels, resize_x, resize_y)
        dummy_output = self._forward_conv(dummy_input)
        self.flattened_size = dummy_output.view(-1).size(0)
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x