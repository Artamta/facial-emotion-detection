from predict import classify_images

# List of image paths to classify
image_paths = [
    './data/anger1.jpg',
    './data/contempt1.jpg',
    './data/disgust1.jpg',
    './data/fear1.jpg',
    './data/happy1.jpg',
    './data/sadness1.jpg',
    './data/surprise1.jpg'
]

# Specify the correct checkpoint path
checkpoint_path = './checkpoints/final_weights.pth_fold1.pth'  # Update this to the correct file

# Run predictions
try:
    predictions = classify_images(image_paths, model_name='VGG19', checkpoint_path=checkpoint_path)
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Predicted Emotion: {pred}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except RuntimeError as e:
    print(f"Model loading error: {e}")