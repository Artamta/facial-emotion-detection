import torch
from PIL import Image
from torchvision import transforms
from model import MyCustomModel

def classify_images(image_paths):
    """Classify a list of image paths."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MyCustomModel.get_model('VGG19').to(device)
    model.load_state_dict(torch.load('./checkpoints/final_weights.pth')['net'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    results = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
            results.append(predicted.item())
    return results