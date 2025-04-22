import torch
from model import MyCustomModel
from dataset import transform
from PIL import Image

def predict_image(image_path, num_classes):
    # Load model
    model = MyCustomModel(num_classes=num_classes)
    model.load_state_dict(torch.load("checkpoints/final_weights.pth"))
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()