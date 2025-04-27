import os
import numpy as np
from PIL import Image
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from config import resize_x, resize_y

class CKDataset(Dataset):
    def __init__(self, split='Training', fold=1, transform=None):
        self.transform = transform
        self.split = split
        self.fold = fold
        self.data = h5py.File('./data/CK_data.h5', 'r', driver='core')

        number = len(self.data['data_label'])
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
            self.data_x = [self.data['data_pixel'][i] for i in train_index]
            self.data_y = [self.data['data_label'][i] for i in train_index]
        else:
            self.data_x = [self.data['data_pixel'][i] for i in test_index]
            self.data_y = [self.data['data_label'][i] for i in test_index]

    def __getitem__(self, index):
        img, label = self.data_x[index], self.data_y[index]
        img = Image.fromarray(np.concatenate((img[:, :, np.newaxis],) * 3, axis=2))
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_x)

def unicornLoader(split, fold, transform, batch_size):
    dataset = CKDataset(split=split, fold=fold, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'Training'))