from predict import classify_images

# List of image paths to classify
image_paths = [
    './data/img01.jpg',
    './data/img02.jpg',
    './data/img03.jpg'
]

# Run predictions
predictions = classify_images(image_paths, model_name='ResNet18', checkpoint_path='./checkpoints/final_weights.pth')

# Print results
for img_path, pred in zip(image_paths, predictions):
    print(f"Image: {img_path}, Predicted Emotion: {pred}")