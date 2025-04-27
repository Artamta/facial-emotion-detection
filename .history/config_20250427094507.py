# Hyperparameters
batch_size = 32  # Number of samples per batch
epochs = 50  # Total number of training epochs
learning_rate = 0.01  # Initial learning rate
learning_rate_decay_start = 20  # Epoch to start decaying the learning rate
learning_rate_decay_every = 1  # Decay the learning rate every 'n' epochs
learning_rate_decay_rate = 0.8  # Factor by which the learning rate is decayed

# Image dimensions
resize_x = 48  # Width of the resized image
resize_y = 48  # Height of the resized image
input_channels = 1  # Number of input channels (1 for grayscale, 3 for RGB)

# Dataset paths
dataset_path = './data/CK_data.h5'  # Path to the CK+ dataset file

# Model configuration
model_name = 'VGG19'  # Choose between 'VGG19' or 'ResNet18'

# Checkpoints
checkpoint_path = './checkpoints/final_weights.pth'  # Path to save/load model weights