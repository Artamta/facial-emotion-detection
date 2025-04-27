import torch
from torch import nn, optim
from model import get_model
from dataset import get_dataloader
from config import batchsize, epochs, learning_rate, dataset_name

def train_model():
    # Initialize model, loss, optimizer
    model = get_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for fold in range(1, 11):  # 10 folds
        print(f"Training fold {fold}")
        train_loader = get_dataloader('Training', fold, batchsize)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Save checkpoint for this fold
        torch.save({'net': model.state_dict()}, f'checkpoints/{dataset_name}_{model_name}_fold_{fold}.pth')