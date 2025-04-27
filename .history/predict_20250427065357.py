import torch
from dataset import get_dataloader
from model import get_model

def predict():
    model = get_model()
    model.load_state_dict(torch.load('./checkpoints/final_weights.pth'))
    model.eval()

    test_loader = get_dataloader('Testing', fold=1, batch_size=1)
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
    return predictions