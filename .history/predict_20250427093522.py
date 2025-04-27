import torch
from PIL import Image
from torchvision import transforms
from models.vgg import VGG
from config import resize_x, resize_y

def predict_image(model, image_path, transform, device):
    """Predict the class of a single image."""
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return predicted.item()

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # Load the model
    model = VGG('VGG19').to(device)
    checkpoint = torch.load('./checkpoints/final_weights.pth', map_location=device)
    model.load_state_dict(checkpoint['net'])
    print("[DEBUG] Model loaded successfully.")

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Predict an image
    image_path = './data/img01.jpg'  # Replace with your image path
    predicted_class = predict_image(model, image_path, transform, device)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(f"[DEBUG] Predicted class: {class_names[predicted_class]}")