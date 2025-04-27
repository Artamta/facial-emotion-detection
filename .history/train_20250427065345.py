import torch
from torch import nn, optim
from dataset import get_dataloader
from model import get_model
from config import batchsize, epochs, learning_rate

def train_model():
    model = get_model()
    train_loader = get_dataloader('Training', fold=1, batch_size=batchsize)
    test_loader = get_dataloader('Testing', fold=1, batch_size=batchsize)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), './checkpoints/final_weights.pth')