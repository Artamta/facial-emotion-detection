import torch
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y

def predict(model, list_of_img_paths, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
    ])
    results = []
    with torch.no_grad():
        for img_path in list_of_img_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            results.append(predicted.item())
    return results