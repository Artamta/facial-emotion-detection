import torch
from PIL import Image
from torchvision import transforms
from model import MyCustomModel

# Define the emotion classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def classify_images(image_paths, model_name='ResNet18', checkpoint_path='./checkpoints/final_weights.pth'):
    """
    Classify a list of image paths using the trained model.

    Args:
        image_paths (list): List of image file paths to classify.
        model_name (str): Name of the model to use ('ResNet18').
        checkpoint_path (str): Path to the saved model checkpoint.

    Returns:
        list: Predicted emotion labels for each image.
    """
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = MyCustomModel.get_model(model_name, num_classes=7).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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