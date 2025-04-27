import os
import torch
from PIL import Image
from torchvision import transforms
from model import MyCustomModel
from config import checkpoint_path

# Define the emotion classes
EMOTION_CLASSES = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

def classify_images(image_paths, model_name='VGG19', model_path='./checkpoints/final_weights.pth_fold4.pth'):
    """
    Classify a list of image paths using a specific model.

    Args:
        image_paths (list): List of image file paths to classify.
        model_name (str): Name of the model to use ('VGG19' or 'ResNet18').
        model_path (str): Path to the specific model checkpoint.

    Returns:
        list: Predicted emotion labels for each image.
    """
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load the specific model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return []
    model = MyCustomModel.get_model(model_name, num_classes=7).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    # Define the transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
        transforms.Resize((48, 48)),  # Resize to match training dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])

    # Classify each image
    results = []
    for image_path in image_paths:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                outputs = model(img)
                _, predicted = outputs.max(1)  # Get the class with the highest score
                results.append(EMOTION_CLASSES[predicted.item()])
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append(None)
    return results

if __name__ == "__main__":
    # Example usage: Predict using the model from fold 4
    model_path = './checkpoints/final_weights.pth_fold4.pth'
    image_paths = [f'./data/img{i}.jpg' for i in range(1, 71)]  # 70 images
    predictions = classify_images(image_paths, model_path=model_path)
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Predicted Emotion: {pred}")