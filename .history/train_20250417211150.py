import torch
from torch import nn, optim
from model import MyCustomModel
from dataset import get_dataloader
from config import epochs, learning_rate

def train_model(data_dir, num_classes):
    # Load data
    _, train_loader = get_dataloader(data_dir)

    # Initialize model
    model = MyCustomModel(num_classes=num_classes)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Save model weights
    torch.save(model.state_dict(), "checkpoints/final_weights.pth")
    print("Model saved to checkpoints/final_weights.pth")