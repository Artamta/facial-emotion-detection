import torch
import torch.optim as optim
import torch.nn as nn
from config import batch_size, epochs, learning_rate, checkpoint_path, data_augmentation, resize_x, resize_y, num_folds,dataset_path
from dataset import get_dataloader
from model import MyCustomModel
from torchvision import transforms

def evaluate_model(model, data_loader, loss_fn, device):
    """
    Evaluate the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for evaluation.
        loss_fn (torch.nn.Module): The loss function.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The accuracy of the model on the given data loader.
    """
    model.eval()
    correct, total, test_loss = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy

def train_model(fold):
    """
    Train the model for a specific fold.

    Args:
        fold (int): The fold number for cross-validation.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation for training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize_x, resize_y)),
        transforms.RandomHorizontalFlip(p=data_augmentation["random_horizontal_flip"]),
        transforms.RandomRotation(data_augmentation["random_rotation"]),
        transforms.RandomAffine(
            degrees=data_augmentation["random_affine"]["degrees"],
            translate=data_augmentation["random_affine"]["translate"]
        ),
        transforms.ColorJitter(
            brightness=data_augmentation["color_jitter"]["brightness"],
            contrast=data_augmentation["color_jitter"]["contrast"],
            saturation=data_augmentation["color_jitter"]["saturation"],
            hue=data_augmentation["color_jitter"]["hue"]
        ),
        transforms.RandomCrop(
            size=data_augmentation["random_crop"]["size"],
            padding=data_augmentation["random_crop"]["padding"]
        ),
        transforms.RandomPerspective(
            distortion_scale=data_augmentation["random_perspective"]["distortion_scale"],
            p=data_augmentation["random_perspective"]["p"]
        ),
        transforms.ToTensor(),  # Convert to tensor before applying RandomErasing
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(
            p=data_augmentation["random_erasing"]["p"],
            scale=data_augmentation["random_erasing"]["scale"],
            ratio=data_augmentation["random_erasing"]["ratio"]
        )
    ])
    train_loader = get_dataloader(split='Training', fold=fold, transform=transform, batch_size=batch_size)

    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_loader = get_dataloader(split='Testing', fold=fold, transform=test_transform, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = MyCustomModel.get_model('VGG19', num_classes=7).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

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
        print(f"Fold {fold}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Evaluate on the test set
        test_acc = evaluate_model(model, test_loader, loss_fn, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({'net': model.state_dict()}, f"{checkpoint_path}_fold{fold}.pth")
            print(f"New best test accuracy for fold {fold}: {best_test_acc:.2f}%")

    print(f"Training completed for fold {fold}. Best Test Accuracy: {best_test_acc:.2f}%")
    return best_test_acc

if __name__ == "__main__":
    fold_accuracies = []

    for fold in range(1, num_folds + 1):
        print(f"Starting training for fold {fold}/{num_folds}...")
        fold_acc = train_model(fold)
        fold_accuracies.append(fold_acc)

    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Cross-validation completed. Average Test Accuracy: {avg_accuracy:.2f}%")