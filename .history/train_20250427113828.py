import torch
import torch.optim as optim
import torch.nn as nn
from config import batch_size, epochs, learning_rate, checkpoint_path, data_augmentation
from dataset import get_dataloader
from model import MyCustomModel
from torchvision import transforms

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        transforms.RandomErasing(
            p=data_augmentation["random_erasing"]["p"],
            scale=data_augmentation["random_erasing"]["scale"],
            ratio=data_augmentation["random_erasing"]["ratio"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_loader = get_dataloader(split='Training', fold=1, transform=transform, batch_size=batch_size)

    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_loader = get_dataloader(split='Testing', fold=1, transform=test_transform, batch_size=batch_size)

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
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Evaluate on the test set
        model.eval()
        test_acc = evaluate_model(model, test_loader, loss_fn, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({'net': model.state_dict()}, checkpoint_path)
            print(f"New best test accuracy: {best_test_acc:.2f}%")

    print(f"Training completed. Best Test Accuracy: {best_test_acc:.2f}%")

if __name__ == "__main__":
    train_model()