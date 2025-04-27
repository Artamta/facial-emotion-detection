# Hyperparameters
batch_size = 128  # Number of samples per batch
epochs = 30  # Total number of training epochs
learning_rate = 0.01  # Initial learning rate
learning_rate_decay_start = 20  # Epoch to start decaying the learning rate
learning_rate_decay_every = 1  # Decay the learning rate every 'n' epochs
learning_rate_decay_rate = 0.8  # Factor by which the learning rate is decayed

# Image dimensions
resize_x = 48  # Width of the resized image
resize_y = 48  # Height of the resized image
input_channels = 1  # Number of input channels (1 for grayscale, 3 for RGB)

# Dataset paths
dataset_path = '/Users/ayush/Desktop/project_ayush_raj/data/CK_data.h5'

# Model configuration
model_name = 'VGG19'

# Checkpoints
checkpoint_path = './checkpoints/final_weights.pth'

# Data augmentation configuration
data_augmentation = {
    "random_horizontal_flip": 0.5,  # Probability of horizontal flip
    "random_rotation": 15,  # Rotate images by ±15 degrees
    "random_affine": {"degrees": 0, "translate": (0.2, 0.2)},  # Translate images by up to 20%
    "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1},  # Adjust brightness, contrast, etc.
    "random_crop": {"size": (40, 40), "padding": 4},  # Randomly crop images with padding
    "random_perspective": {"distortion_scale": 0.5, "p": 0.5},  # Apply perspective distortion
    "random_erasing": {"p": 0.5, "scale": (0.02, 0.33), "ratio": (0.3, 3.3)},  # Randomly erase parts of the image
}