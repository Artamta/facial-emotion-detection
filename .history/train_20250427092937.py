import time
import torch
import torch.optim as optim
from config import epochs, learning_rate, learning_rate_decay_start, learning_rate_decay_every, learning_rate_decay_rate
from dataset import unicornLoader
from utils import set_lr, clip_gradient, progress_bar

def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[DEBUG] Model moved to device:", device)

    for epoch in range(num_epochs):
        print(f"[DEBUG] Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        if epoch > learning_rate_decay_start:
            decay_factor = learning_rate_decay_rate ** ((epoch - learning_rate_decay_start) // learning_rate_decay_every)
            current_lr = learning_rate * decay_factor
            set_lr(optimizer, current_lr)
        else:
            current_lr = learning_rate

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"[DEBUG] Processing batch {batch_idx + 1}/{len(train_loader)}")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            print(f"[DEBUG] Loss for batch {batch_idx + 1}: {loss.item():.4f}")
            loss.backward()
            clip_gradient(optimizer, 0.1)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx + 1, len(train_loader), msg=f"Loss: {loss.item():.4f}")

        print(f"[DEBUG] Epoch {epoch + 1} completed. Loss: {total_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")