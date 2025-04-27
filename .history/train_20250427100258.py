import torch
import torch.optim as optim
import torch.nn as nn
from config import batch_size, epochs, learning_rate
from dataset import get_dataloader
from model import MyCustomModel
from torchvision import transforms

def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_loader = get_dataloader(split='Training', fold=1, transform=transform, batch_size=batch_size)
    test_loader = get_dataloader(split='Testing', fold=1, transform=transform, batch_size=batch_size)

    model = MyCustomModel.get_model('VGG19').to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train_model()