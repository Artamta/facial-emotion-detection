import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from config import dataset_path, resize_x, resize_y

class CustomDataset(Dataset):
    """Custom Dataset for loading data."""
    def __init__(self, split='Training', fold=1, transform=None):
        self.transform = transform
        self.split = split
        self.fold = fold

        # Load the dataset
        with h5py.File(dataset_path, 'r') as data:
            self.data_x = np.array(data['data_pixel'])
            self.data_y = np.array(data['data_label'])

        # Split the dataset into training and testing sets
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

        if self.split == 'Training':
            self.data_x = [self.data_x[i] for i in train_index]
            self.data_y = [self.data_y[i] for i in train_index]
        else:
            self.data_x = [self.data_x[i] for i in test_index]
            self.data_y = [self.data_y[i] for i in test_index]

    def __getitem__(self, index):
        img, label = self.data_x[index], self.data_y[index]
        img = Image.fromarray(img).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_x)

def get_dataloader(split, fold, transform, batch_size):
    dataset = CustomDataset(split=split, fold=fold, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'Training'))