import torch
import torch.optim as optim
import torch.nn as nn
from config import batch_size, epochs, learning_rate, learning_rate_decay_start, learning_rate_decay_every, learning_rate_decay_rate
from dataset import get_dataloader
from utils import set_lr, clip_gradient, progress_bar
from models.vgg import VGG

def train_model(model, num_epochs, train_loader, test_loader, loss_fn, optimizer, device):
    """Train the model and evaluate on the test set."""
    best_test_acc = 0

    for epoch in range(num_epochs):
        print(f"\n[DEBUG] Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Adjust learning rate
        if epoch > learning_rate_decay_start:
            decay_factor = learning_rate_decay_rate ** ((epoch - learning_rate_decay_start) // learning_rate_decay_every)
            current_lr = learning_rate * decay_factor
            set_lr(optimizer, current_lr)
        else:
            current_lr = learning_rate
        print(f"[DEBUG] Current learning rate: {current_lr}")

        # Training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            clip_gradient(optimizer, grad_clip=0.1)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx + 1, len(train_loader), msg=f"Loss: {loss.item():.4f}")

        train_acc = 100. * correct / total
        print(f"[DEBUG] Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Evaluate on the test set
        test_acc = evaluate_model(model, test_loader, loss_fn, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"[DEBUG] New best test accuracy: {best_test_acc:.2f}%")

    print(f"[DEBUG] Training completed. Best Test Accuracy: {best_test_acc:.2f}%")

def evaluate_model(model, test_loader, loss_fn, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total
    print(f"[DEBUG] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_acc

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # Load dataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_loader = get_dataloader(split='Training', fold=1, transform=transform, batch_size=batch_size)
    test_loader = get_dataloader(split='Testing', fold=1, transform=transform, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = VGG('VGG19').to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Train the model
    train_model(model, epochs, train_loader, test_loader, loss_fn, optimizer, device)