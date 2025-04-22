import torch
import torch.nn as nn
import torch.nn.functional as F
import config # Import config to get image dimensions

# Define the VGG16 model from scratch
class VGG16_Scratch(nn.Module):
    def __init__(self, num_classes=1000): # Default to 1000, will be set during instantiation
        super(VGG16_Scratch, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Use AdaptiveAvgPool2d with (1, 1) output for compatibility
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate the flattened size after pooling (512 channels * 1 * 1)
        self.flattened_size = 512 * 1 * 1
        # print(f"Calculated flattened size for FC layer (VGG16): {self.flattened_size}") # Optional print

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.classifier(x)
        return x


# Define a Simple CNN model for quick testing
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=config.input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Input: (batch_size, 3, 128, 128) -> Output: (batch_size, 16, 128, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (batch_size, 16, 64, 64)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Input: (batch_size, 16, 64, 64) -> Output: (batch_size, 32, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: (batch_size, 32, 32, 32)

        # Calculate the flattened size dynamically based on config image size
        # After conv1, pool1 -> size = 128 / 2 = 64
        # After conv2, pool2 -> size = 64 / 2 = 32
        self._calculate_fc_input_size()

        # Fully Connected Layer
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes) # Output layer

    def _calculate_fc_input_size(self):
        # Create a dummy tensor with the input size
        # Use config values for height and width
        dummy_input = torch.randn(1, config.input_channels, config.resize_height, config.resize_width)
        # Pass it through the convolutional/pooling layers
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        # Calculate the number of features after flattening
        self.fc_input_size = x.numel() // x.shape[0] # numel() gives total elements, divide by batch size (1)
        print(f"Calculated flattened size for FC layer (SimpleCNN): {self.fc_input_size}")


    def forward(self, x):
        # Pass through conv layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1) # Flatten all dimensions except batch

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No activation here, CrossEntropyLoss will apply softmax
        return x

# --- Keep VGG16 commented out or remove if not needed for now ---
# class VGG16_Scratch(nn.Module):
#     # ... (previous VGG16 code) ...

# --- Update the alias to point to the SimpleCNN ---
EmotionVGGModel = SimpleCNN # Use SimpleCNN for testing
# EmotionVGGModel = VGG16_Scratch # Switch back to this for final training