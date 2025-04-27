import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from config import resize_x, resize_y

class CKDataset(Dataset):
    def __init__(self, split, fold, transform=None):
        self.split = split
        self.fold = fold
        self.transform = transform
        self.data_dir = f"data/{split}/fold_{fold}"  # Example directory structure
        self.image_paths = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(img_path.split('_')[-1][0])  # Example: extract label from filename
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataloader
def get_dataloader(split, fold, batch_size):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
    ])
    dataset = CKDataset(split=split, fold=fold, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader