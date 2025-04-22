import torch

# --- Device Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Dataset Configuration ---
# IMPORTANT: Update this path if your data directory location changes
DATASET_PATH = '/Users/ayush/Desktop/project_ayush_raj/data'
# Image dimensions
RESIZE_H = 128
RESIZE_W = 128
INPUT_CHANNELS = 3 # Assuming RGB images

# --- Training Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # Number of epochs to train for

# --- Model Saving ---
CHECKPOINT_DIR = '/Users/ayush/Desktop/project_ayush_raj/checkpoints'
FINAL_WEIGHTS_FILE = 'final_weights.pth'