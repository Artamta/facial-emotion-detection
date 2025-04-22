import torch
import torch.nn.functional as F
from PIL import Image
import os
import config # Import config variables
from model import EmotionVGGModel # Import your model class
from dataset import data_transform # Import the same transform used for training

# Global variable to hold the loaded model and class names
loaded_model = None
class_names = None

def load_model_for_inference(num_classes):
    """Loads the trained model from the checkpoint."""
    global loaded_model
    model = EmotionVGGModel(num_classes=num_classes)
    try:
        # Load state dict, ensuring it's mapped to the correct device
        model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
        model.to(config.device)
        model.eval() # Set to evaluation mode
        print(f"Model loaded successfully from {config.checkpoint_path}")
        loaded_model = model
        return model
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {config.checkpoint_path}")
        return None
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return None

def predict_single_image(image_path, model, classes, device, transform):
    """Predicts the class for a single image file."""
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, None

    if model is None or classes is None:
        print("Error: Model or class names not loaded.")
        return None, None

    try:
        # Load and transform the image
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device) # Add batch dim and move to device

        # Perform inference
        with torch.no_grad():
            outputs = model(batch_t)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_label = classes[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_label, confidence_score

    except Exception as e:
        print(f"An error occurred during prediction for {image_path}: {e}")
        return None, None

# Function expected by interface.py
# It needs access to class_names, which might require loading the dataset first
# or storing class_names somewhere accessible (e.g., a file).
# For simplicity here, we'll assume class_names are loaded when the model is loaded.
# A more robust solution might involve passing class_names or loading them separately.
def run_prediction_on_image(image_path):
    """
    Loads the model (if not already loaded), gets class names (needs improvement),
    and runs prediction on the specified image path.
    """
    global loaded_model, class_names

    # --- This part needs refinement ---
    # Ideally, class_names should be saved during training or loaded reliably.
    # Here, we try to infer them by loading the dataset structure again.
    if class_names is None:
        try:
            temp_dataset = datasets.ImageFolder(root=config.dataset_path) # No transform needed just for classes
            class_names = temp_dataset.classes
            print(f"Inferred class names: {class_names}")
        except Exception as e:
            print(f"Could not infer class names from dataset structure: {e}")
            return None, None # Cannot predict without class names
    # --- End refinement section ---

    if loaded_model is None:
        if class_names:
            load_model_for_inference(num_classes=len(class_names))
        else:
            print("Cannot load model without knowing the number of classes.")
            return None, None

    if loaded_model is None:
         print("Model could not be loaded. Cannot predict.")
         return None, None

    # Now run the prediction
    return predict_single_image(image_path, loaded_model, class_names, config.device, data_transform)


# Define the name expected by interface.py
emotion_predictor_function = run_prediction_on_image

# Allow running prediction directly (example)
if __name__ == '__main__':
    # IMPORTANT: Replace with a valid image path from your data/ directory or elsewhere
    test_img = '/Users/ayush/Desktop/project_ayush_raj/data/train/angry' # Example path - CHANGE THIS

    if not os.path.exists(test_img):
         print(f"Test image '{test_img}' not found. Please provide a valid path.")
    else:
        print(f"\n--- Testing Prediction on: {test_img} ---")
        predicted_class, confidence = emotion_predictor_function(test_img)

        if predicted_class is not None:
            print(f"--> Predicted Emotion: {predicted_class} (Confidence: {confidence:.2f})")
        else:
            print("--> Prediction failed.")