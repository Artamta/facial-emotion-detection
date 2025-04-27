# Hyperparameters for training
batch_size = 32  # Number of samples per batch
epochs = 60  # Total number of epochs for training
learning_rate = 0.01  # Initial learning rate
checkpoint_path = './checkpoints/final_weights.pth'  # Path to save model weights

# Dataset configuration
dataset_path = './data/CK_data.h5'  # Path to the dataset file

# Image dimensions
resize_x = 48  # Width of the resized image
resize_y = 48  # Height of the resized image
input_channels = 1  # Number of input channels (1 for grayscale, 3 for RGB)

# Data augmentation configuration
data_augmentation = {
    "random_horizontal_flip": 0.5,  # Probability of horizontal flip
    "random_rotation": 15,  # Maximum rotation angle in degrees
    "random_affine": {"degrees": 0, "translate": (0.2, 0.2)},  # Affine transformation
    "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1},  # Color jitter
    "random_crop": {"size": (40, 40), "padding": 4},  # Random cropping with padding
    "random_perspective": {"distortion_scale": 0.5, "p": 0.5},  # Perspective distortion
    "random_erasing": {"p": 0.5, "scale": (0.02, 0.33), "ratio": (0.3, 3.3)},  # Random erasing
}

# Cross-validation configuration
num_folds = 5  # Number of folds for k-fold cross-validation