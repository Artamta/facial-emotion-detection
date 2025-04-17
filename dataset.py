import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import resize_x, resize_y, batchsize

# Define transformations
transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor()
])

# Custom Dataset
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

# Dataloader
def get_dataloader(data_dir):
    dataset = CustomDataset(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return dataset, dataloader