import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from config import batch_size, epochs, learning_rate, checkpoint_path, num_folds
from dataset import get_dataloader
from model import MyCustomModel
from torchvision import transforms

def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
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

def train_model():
    """
    Train the model and evaluate it on the test set for each fold.
    """
    # Check for MPS, CUDA, or fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    fold_accuracies = []

    for fold in range(1, num_folds + 1):
        print(f"\n[DEBUG] Starting training for fold {fold}/{num_folds}")

        train_loader = get_dataloader(split='Training', fold=fold, transform=transform, batch_size=batch_size)
        test_loader = get_dataloader(split='Testing', fold=fold, transform=transform, batch_size=batch_size)

        model = MyCustomModel.get_model(model_name='VGG19', num_classes=7).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        
        # Initialize the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)  # Reduce LR by 10x every 20 epochs
        
        best_test_acc = 0
        for epoch in range(epochs):
            print(f"\n[DEBUG] Starting epoch {epoch + 1}/{epochs}")
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total
            print(f"[DEBUG] Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

            test_acc = evaluate_model(model, test_loader, loss_fn, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({'net': model.state_dict()}, f"{checkpoint_path}_fold{fold}.pth")
                print(f"[DEBUG] Model weights saved for fold {fold} to {checkpoint_path}_fold{fold}.pth")

            # Step the scheduler at the end of the epoch
            scheduler.step()
            print(f"[DEBUG] Learning rate for next epoch: {scheduler.get_last_lr()}")

        print(f"[DEBUG] Training completed for fold {fold}. Best Test Accuracy: {best_test_acc:.2f}%")
        fold_accuracies.append(best_test_acc)

    # Calculate and print the average accuracy across all folds
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\n[DEBUG] Cross-validation completed. Average Test Accuracy: {avg_accuracy:.2f}%")

def test_train_model():
    """
    Test the train_model function to ensure it works as expected.
    """
    print("\n[TEST] Starting test for train_model...")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Get a small dataloader for testing
    train_loader = get_dataloader(split='Training', fold=1, transform=transform, batch_size=batch_size)
    test_loader = get_dataloader(split='Testing', fold=1, transform=transform, batch_size=batch_size)

    # Instantiate the model, loss function, and optimizer
    model = MyCustomModel.get_model(model_name='VGG19', num_classes=7)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Run a single epoch for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"[TEST] Single batch processed. Loss: {loss.item():.4f}")
        break

    # Evaluate the model
    evaluate_model(model, test_loader, loss_fn, device)
    print("[TEST] train_model test completed successfully.")

if __name__ == "__main__":
    # Uncomment the following line to run the full training
    # train_model()

    # Run the test for train_model
    test_train_model()