import torch
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y

def predict(model, image_paths, device):
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)

    batch = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(batch)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().numpy()