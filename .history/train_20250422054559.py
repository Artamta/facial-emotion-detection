import torch
import torch.nn as nn
import torch.optim as optim
import config # Import config variables
from model import EmotionVGGModel # Import your model class
from dataset import emotion_dataset_creator, emotion_dataloader_creator, data_transform # Import dataset functions and transform
import os

def run_training():
    """Runs the training loop and saves the final model."""
    print(f"Using device: {config.device}")

    # 1. Load Dataset and DataLoader
    try:
        train_dataset = emotion_dataset_creator(config.dataset_path, data_transform)
        train_loader, class_names = emotion_dataloader_creator(train_dataset, config.batch_size, shuffle=True)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return # Exit if data loading fails

    if not train_loader or not class_names:
        print("DataLoader or class names not available. Exiting training.")
        return

    num_classes = len(class_names)
    print(f"Number of classes detected: {num_classes}")

    # 2. Initialize Model, Loss, Optimizer
    model = EmotionVGGModel(num_classes=num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\nModel Structure:")
    # print(model) # Optional: Print model structure
    print("Starting training...")

    # 3. Training Loop
    for epoch in range(config.num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        processed_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.device), labels.to(config.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            processed_samples += images.size(0)

            # Print progress (optional)
            if (i + 1) % 10 == 0: # Print every 10 batches
                 print(f'Epoch [{epoch+1}/{config.num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / processed_samples
        print(f"Epoch {epoch+1}/{config.num_epochs}, Average Loss: {epoch_loss:.4f}")

    print("Training finished.")

    # 4. Save the final model weights
    try:
        torch.save(model.state_dict(), config.checkpoint_path)
        print(f"Model state dictionary saved to {config.checkpoint_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Define the name expected by interface.py
emotion_trainer_function = run_training

# Allow running training directly from this script
if __name__ == '__main__':
    emotion_trainer_function()