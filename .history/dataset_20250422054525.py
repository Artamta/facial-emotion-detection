import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config # Import variables from config.py
import os

# Define transformations using config values
data_transform = transforms.Compose([
    transforms.Resize((config.resize_height, config.resize_width)),
    transforms.ToTensor(),
    # Add normalization if you calculated and used mean/std
    # transforms.Normalize(mean=[...], std=[...])
])

# Function to create the dataset
def create_dataset(data_path, transform):
    """Creates an ImageFolder dataset."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found at: {data_path}")
    try:
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        print(f"Dataset loaded successfully from {data_path}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Class names: {dataset.classes}")
        print(f"Number of samples: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        raise # Re-raise the exception after printing

# Function to create the DataLoader
def create_dataloader(dataset, batch_size, shuffle=True):
    """Creates a DataLoader for the given dataset."""
    if dataset is None:
        print("Dataset is None, cannot create DataLoader.")
        return None, None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    class_names = dataset.classes if hasattr(dataset, 'classes') else None
    return loader, class_names

# Example of how to use these functions (optional, for testing this file)
if __name__ == '__main__':
    print("Testing dataset loading...")
    try:
        train_dataset = create_dataset(config.dataset_path, data_transform)
        if train_dataset:
            train_loader, classes = create_dataloader(train_dataset, config.batch_size)
            if train_loader:
                print(f"DataLoader created. Found classes: {classes}")
                # You could add code here to fetch and show a batch if needed
            else:
                print("Failed to create DataLoader.")
        else:
            print("Failed to create Dataset.")
    except Exception as e:
        print(f"Error during dataset test: {e}")

# Define names expected by interface.py
EmotionImageDataset = datasets.ImageFolder # Using the standard ImageFolder
emotion_dataloader_creator = create_dataloader # Function to create loader
emotion_dataset_creator = create_dataset # Function to create dataset