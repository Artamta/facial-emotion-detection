# --- Model ---
# Replace 'EmotionVGGModel' with the actual name of your model class in model.py
from model import EmotionVGGModel as TheModel

# --- Training ---
# Replace 'emotion_trainer_function' with the actual name of your training function in train.py
from train import emotion_trainer_function as the_trainer

# --- Prediction ---
# Replace 'emotion_predictor_function' with the actual name of your prediction function in predict.py
from predict import emotion_predictor_function as the_predictor

# --- Dataset ---
# Replace 'EmotionImageDataset' with your actual Dataset class/object name in dataset.py
# If using ImageFolder directly, you might need a wrapper or adjust how it's used.
# Using the creator functions defined in dataset.py might be more flexible.
# from dataset import EmotionImageDataset as TheDataset # Option 1: If you have a custom class
from dataset import emotion_dataset_creator as TheDatasetCreator # Option 2: Use the creator function

# --- DataLoader ---
# Replace 'emotion_dataloader_creator' with your actual dataloader function/object name in dataset.py
from dataset import emotion_dataloader_creator as the_dataloader_creator # Use the creator function

# --- Configuration ---
# These should match the variable names in your config.py
from config import batch_size as the_batch_size
from config import num_epochs as total_epochs
from config import learning_rate as the_learning_rate # Example: if needed by grader
from config import resize_height as the_resize_height # Example: if needed by grader
from config import resize_width as the_resize_width # Example: if needed by grader
from config import device as the_device # Example: if needed by grader
from config import dataset_path as the_dataset_path # Example: if needed by grader
from config import checkpoint_path as the_checkpoint_path # Example: if needed by grader

print("Interface mappings loaded.")
# You can add checks here to ensure imports worked if desired