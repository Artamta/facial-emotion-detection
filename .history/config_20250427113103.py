# Hyperparameters
batch_size = 128 # Number of samples per batch
epochs = 30  # Total number of training epochs
learning_rate = 0.01  # Initial learning rate
learning_rate_decay_start = 20  # Epoch to start decaying the learning rate
learning_rate_decay_every = 1  # Decay the learning rate every 'n' epochs
learning_rate_decay_rate = 0.8  # Factor by which the learning rate is decayed

# Image dimensions
resize_x = 48  # Width of the resized image
resize_y = 48  # Height of the resized image
input_channels = 1  # Grayscale images

# Dataset paths
dataset_path = '/Users/ayush/Desktop/project_ayush_raj/data/CK_data.h5'

# Model configuration
model_name = 'VGG19'

# Checkpoints
checkpoint_path = './checkpoints/final_weights.pth'