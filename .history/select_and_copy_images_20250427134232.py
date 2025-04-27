import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MyCustomModel

# Define the emotion classes
EMOTION_CLASSES = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

# List of model checkpoint paths
checkpoint_paths = [
    './checkpoints/final_weights.pth_fold1.pth',
    './checkpoints/final_weights.pth_fold2.pth',
    './checkpoints/final_weights.pth_fold3.pth',
    './checkpoints/final_weights.pth_fold4.pth',
    './checkpoints/final_weights.pth_fold5.pth'
]

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale input
    transforms.Resize((48, 48)),  # Resize to match training dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

def load_models(checkpoint_paths, model_name='VGG19', num_classes=7, device='cpu'):
    """Load all models from the checkpoint paths."""
    models = []
    for path in checkpoint_paths:
        model = MyCustomModel.get_model(model_name=model_name, num_classes=num_classes).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        models.append(model)
    return models

def ensemble_predict(models, image_paths, device='cpu'):
    """Make predictions using an ensemble of models."""
    predictions = []
    for img_path in image_paths:
        # Load and preprocess the image
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Collect predictions from all models
        model_outputs = []
        with torch.no_grad():
            for model in models:
                outputs = model(input_tensor)
                model_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())

        # Average the probabilities across all models
        avg_output = np.mean(model_outputs, axis=0)
        predicted_class = np.argmax(avg_output, axis=1)[0]
        predictions.append(EMOTION_CLASSES[predicted_class])

    return predictions

if __name__ == "__main__":
    # List of image paths to classify
    image_paths = [
        './data/image1.jpg',
        './data/image2.jpg',
        './data/image3.jpg',
        './data/image4.jpg',
        './data/image5.jpg',
        './data/image6.jpg',
        './data/image7.jpg',
        './data/image8.jpg',
        './data/image9.jpg',
        './data/image10.jpg'
    ]

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all models
    models = load_models(checkpoint_paths, device=device)

    # Run ensemble predictions
    predictions = ensemble_predict(models, image_paths, device=device)

    # Print predictions
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Predicted Emotion: {pred}")