import os
import torch
from PIL import Image
from torchvision import transforms
from model import MyCustomModel
from config import checkpoint_path, num_folds

# Define the emotion classes
EMOTION_CLASSES = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

def combine_models(model_name='VGG19', num_classes=7, output_path='./checkpoints/final_combined_model.pth'):
    """
    Combine the weights of all k-fold models into a single final model.

    Args:
        model_name (str): Name of the model architecture (e.g., 'VGG19').
        num_classes (int): Number of output classes.
        output_path (str): Path to save the combined model weights.
    """
    # Initialize the model
    model = MyCustomModel.get_model(model_name=model_name, num_classes=num_classes)
    combined_state_dict = None
    fold_count = 0

    # Iterate through all k-fold weight files
    for fold in range(1, num_folds + 1):
        fold_weight_path = f"{checkpoint_path}_fold{fold}.pth"
        if not os.path.exists(fold_weight_path):
            print(f"[ERROR] Weight file not found: {fold_weight_path}")
            return
        print(f"[INFO] Loading weights from: {fold_weight_path}")
        checkpoint = torch.load(fold_weight_path, map_location='cpu')
        state_dict = checkpoint['net']

        # Add the weights to the combined state dict
        if combined_state_dict is None:
            combined_state_dict = {key: value.clone().float() for key, value in state_dict.items()}
        else:
            for key in combined_state_dict:
                combined_state_dict[key] += state_dict[key].float()

        fold_count += 1

    # Average the weights
    for key in combined_state_dict:
        combined_state_dict[key] /= fold_count

    # Save the combined weights
    torch.save({'net': combined_state_dict}, output_path)
    print(f"[INFO] Combined model saved to: {output_path}")

def classify_images(image_paths, model_name='VGG19', combined_model_path='./checkpoints/final_combined_model.pth'):
    """
    Classify a list of image paths using the combined model.

    Args:
        image_paths (list): List of image file paths to classify.
        model_name (str): Name of the model to use ('VGG19' or 'ResNet18').
        combined_model_path (str): Path to the combined model checkpoint.

    Returns:
        list: Predicted emotion labels for each image.
    """
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load the combined model
    if not os.path.exists(combined_model_path):
        print(f"[ERROR] Combined model file not found: {combined_model_path}")
        return []
    model = MyCustomModel.get_model(model_name, num_classes=7).to(device)
    checkpoint = torch.load(combined_model_path, map_location=device)
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
    Combine k-fold models into a single model
    combined_model_path = './checkpoints/final_combined_model.pth'
    combine_models(output_path=combined_model_path)

    # Example usage: Predict using the combined model
    image_paths = [f'./data/image{i}.jpg' for i in range(1, 71)]  # 70 images
    predictions = classify_images(image_paths, combined_model_path=combined_model_path)
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Predicted Emotion: {pred}")