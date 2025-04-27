import torch
from model import get_model
from dataset import get_dataloader

def predict(fold):
    # Load model
    model = get_model()
    checkpoint_path = f'checkpoints/CK+_VGG19_fold_{fold}.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['net'])
    model.eval()

    # Prepare data
    test_loader = get_dataloader('Testing', fold, batch_size=1)

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
    return predictions