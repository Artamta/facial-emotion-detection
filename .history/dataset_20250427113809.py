import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from config import dataset_path, resize_x, resize_y

class CustomDataset(Dataset):
    """Custom Dataset for loading data from an HDF5 file."""
    def __init__(self, split='Training', fold=10, transform=None):
        """
        Args:
            split (str): 'Training' or 'Testing' to specify the dataset split.
            fold (int): Fold number for k-fold cross-validation (1 to 10).
            transform (callable, optional): Transformations to apply to the images.
        """
        self.transform = transform
        self.split = split
        self.fold = fold

        # Load the dataset
        try:
            with h5py.File(dataset_path, 'r') as data:
                self.data_x = np.array(data['data_pixel'])
                self.data_y = np.array(data['data_label'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please ensure the file exists.")

        # Split the dataset into training and testing sets
        self.train_index, self.test_index = self._split_indices()

        # Select the appropriate data split
        if self.split == 'Training':
            self.data_x = [self.data_x[i] for i in self.train_index]
            self.data_y = [self.data_y[i] for i in self.train_index]
        elif self.split == 'Testing':
            self.data_x = [self.data_x[i] for i in self.test_index]
            self.data_y = [self.data_y[i] for i in self.test_index]
        else:
            raise ValueError("Invalid split. Choose either 'Training' or 'Testing'.")

    def _split_indices(self):
        """Split the dataset into training and testing indices based on the fold."""
        number = len(self.data_y)
        sum_number = [0, 135, 312, 387, 594, 678, 927, 981]
        test_number = [12, 18, 9, 21, 9, 24, 6]

        test_index = []
        train_index = []

        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10:
                    test_index.append(sum_number[j] + (self.fold - 1) * test_number[j] + k)
                else:
                    test_index.append(sum_number[j + 1] - 1 - k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        return train_index, test_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the data item.

        Returns:
            tuple: (image, label) where image is a transformed PIL Image and label is the class index.
        """
        img, label = self.data_x[index], self.data_y[index]
        img = Image.fromarray(img).convert('L')  # Convert to grayscale
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
        fold (int): Fold number for k-fold cross-validation (1 to 10).
        transform (callable): Transformations to apply to the images.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset split.
    """
    dataset = CustomDataset(split=split, fold=fold, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'Training'))