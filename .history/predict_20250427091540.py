import torch
from PIL import Image
from config import resize_x, resize_y

def classify_frogs(model, list_of_img_paths, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    for img_path in list_of_img_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
            results.append(predicted.item())
    return results