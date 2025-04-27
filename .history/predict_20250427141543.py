import torch
from PIL import Image
from torchvision import transforms
from model import MyCustomModel

# Define the emotion classes
EMOTION_CLASSES = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

def classify_images(image_paths, model_name='VGG19', checkpoint_path='/Users/ayush/Desktop/project_ayush_raj/checkpoints/final_weights.pth_fold4.pth', batch_size=32):
    """
    Classify a batch of image paths using the trained model.

    Args:
        image_paths (list): List of image file paths to classify.
        model_name (str): Name of the model to use ('VGG19' or 'ResNet18').
        checkpoint_path (str): Path to the saved model checkpoint.
        batch_size (int): Number of images to process in a single batch.

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

    # Process images in batches
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        # Preprocess each image in the batch
        for image_path in batch_paths:
            try:
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                img = transform(img)  # Apply transformations
                batch_images.append(img)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                batch_images.append(None)

        # Remove None entries and create a batch tensor
        batch_images = [img for img in batch_images if img is not None]
        if len(batch_images) == 0:
            continue
        batch_tensor = torch.stack(batch_images).to(device)  # Stack images into a batch

        # Perform inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, predicted = outputs.max(1)  # Get the class with the highest score
            results.extend([EMOTION_CLASSES[p.item()] for p in predicted])

    return results

if __name__ == "__main__":
    # Example usage
    image_paths = [
        './data/img01.jpg',
        './data/img02.jpg',
        './data/img03.jpg',
        './data/img04.jpg',
        './data/img05.jpg',
        './data/img06.jpg',
        './data/img07.jpg',
        './data/img08.jpg',
        './data/img09.jpg',
        './data/img10.jpg'
    ]
    predictions = classify_images(image_paths, batch_size=4)  # Process in batches of 4
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Predicted Emotion: {pred}")