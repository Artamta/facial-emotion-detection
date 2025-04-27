import torch
from PIL import Image
from torchvision import transforms
from model import MyCustomModel

def classify_images(image_paths):
    """Classify a list of image paths."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = MyCustomModel.get_model('VGG19').to(device)  # Change 'VGG19' to 'ResNet18' if needed
    model.load_state_dict(torch.load('./checkpoints/final_weights.pth')['net'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((48, 48)),  # Ensure this matches your training image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust if using RGB images
    ])

    results = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
            results.append(predicted.item())
    return results