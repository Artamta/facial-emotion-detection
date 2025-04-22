import torch

# Training Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10 # Or your desired number of epochs

# Image Configuration
resize_height = 128
resize_width = 128
input_channels = 3 # Assuming RGB images

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Paths
# Use relative paths for portability
dataset_path = 'data/' # Path to the main data directory containing class subfolders
checkpoint_dir = 'checkpoints/'
final_model_name = 'final_weights.pth'
checkpoint_path = checkpoint_dir + final_model_name

# Ensure checkpoint directory exists
import os
os.makedirs(checkpoint_dir, exist_ok=True)