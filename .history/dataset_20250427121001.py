from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
from torchvision import transforms
from config import dataset_path, data_augmentation, resize_x, resize_y


class CK(data.Dataset):
    """`CK+ Dataset with dynamic data augmentation."""

    def __init__(self, split='Training', fold=1, transform=None):
        self.transform = transform
        self.split = split  # 'Training' or 'Testing'
        self.fold = fold  # Fold number for cross-validation

        # Load the dataset
        try:
            self.data = h5py.File(dataset_path, 'r', driver='core')
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please ensure the file exists.")

        number = len(self.data['data_label'])  # Total number of samples (981)
        sum_number = [0, 135, 312, 387, 594, 678, 927, 981]  # Cumulative sum of class counts
        test_number = [12, 18, 9, 21, 9, 24, 6]  # Number of test samples per class

        test_index = []
        train_index = []

        # Create training and testing indices based on the fold
        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10:  # For folds 1-9
                    test_index.append(sum_number[j] + (self.fold - 1) * test_number[j] + k)
                else:  # For the 10th fold
                    test_index.append(sum_number[j + 1] - 1 - k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        print(f"Training samples: {len(train_index)}, Testing samples: {len(test_index)}")

        # Load the training or testing data
        if self.split == 'Training':
            self.data_x = [self.data['data_pixel'][i] for i in train_index]
            self.data_y = [self.data['data_label'][i] for i in train_index]
        elif self.split == 'Testing':
            self.data_x = [self.data['data_pixel'][i] for i in test_index]
            self.data_y = [self.data['data_label'][i] for i in test_index]
        else:
            raise ValueError("Invalid split. Choose either 'Training' or 'Testing'.")

def __getitem__(self, index):
    """
    Args:
        index (int): Index of the data item.

    Returns:
        tuple: (image, label) where image is a transformed PIL Image and label is the class index.
    """
    img, label = self.data_x[index], self.data_y[index]

    # Convert the image to a PIL Image (grayscale)
    img = Image.fromarray(img)

    # Apply transformations
    if self.transform:
        img = self.transform(img)

    return img, label

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_x)


def get_dataloader(split, fold, transform, batch_size):
    """
    Create a DataLoader for the specified dataset split.

    Args:
        split (str): 'Training' or 'Testing' to specify the dataset split.
        fold (int): Fold number for cross-validation.
        transform (callable): Transformations to apply to the images.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset split.
    """
    dataset = CK(split=split, fold=fold, transform=transform)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'Training'))