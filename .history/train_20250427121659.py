import torch
import torch.optim as optim
import torch.nn as nn
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        model = MyCustomModel.get_model('VGG19').to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

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

        print(f"[DEBUG] Training completed for fold {fold}. Best Test Accuracy: {best_test_acc:.2f}%")
        fold_accuracies.append(best_test_acc)

    # Calculate and print the average accuracy across all folds
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\n[DEBUG] Cross-validation completed. Average Test Accuracy: {avg_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()