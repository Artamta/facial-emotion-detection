import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Import configuration variables
from config import DATASET_PATH, RESIZE_H, RESIZE_W, BATCH_SIZE, DEVICE

# Define transformations using config values
data_transform = transforms.Compose([
    transforms.Resize((RESIZE_H, RESIZE_W)),
    transforms.ToTensor(),
    # Add normalization here if you calculated mean/std and want to use it
    # transforms.Normalize(mean=[...], std=[...])
])

# Global variables to store dataset info
image_dataset = None
data_loader = None
class_names = []
num_classes = 0

try:
    # Load dataset using ImageFolder
    image_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transform)

    # Create DataLoader
    data_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get class names and number of classes
    class_names = image_dataset.classes
    num_classes = len(class_names)

    print(f"Dataset loaded successfully from: {DATASET_PATH}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Number of samples: {len(image_dataset)}")
    print(f"DataLoader created with batch size: {BATCH_SIZE}")

except FileNotFoundError:
    print(f"Error: Dataset directory not found at path: {DATASET_PATH}")
    print("Please ensure the 'data' directory exists and contains subdirectories for each class.")
    # Keep variables as None/empty to allow interface import without error
    image_dataset = None
    data_loader = None
    class_names = []
    num_classes = 0
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    image_dataset = None
    data_loader = None
    class_names = []
    num_classes = 0

# You could define a custom Dataset class here if ImageFolder is not sufficient
# class CustomImageDataset(torch.utils.data.Dataset):
#     def __init__(self, ...):
#         # ... implementation ...
#     def __len__(self):
#         # ... implementation ...
#     def __getitem__(self, idx):
#         # ... implementation ...

# Export the necessary components
TheDataset = datasets.ImageFolder # Exporting ImageFolder as the dataset class used
the_dataloader = data_loader      # Exporting the created DataLoader instance