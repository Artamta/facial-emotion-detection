import torch

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)

            # Forward pass
            outputs = model(batch)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")