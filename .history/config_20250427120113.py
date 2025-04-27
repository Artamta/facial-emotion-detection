# Hyperparameters
batch_size = 128
epochs = 30
learning_rate = 0.01
checkpoint_path = './checkpoints/final_weights.pth'

# Dataset path
dataset_path = './data/CK_data.h5'

# Image dimensions
resize_x = 48
resize_y = 48
input_channels = 1  # Grayscale images

# Data augmentation configuration
data_augmentation = {
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "random_affine": {"degrees": 0, "translate": (0.2, 0.2)},
    "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1},
    "random_crop": {"size": (40, 40), "padding": 4},
    "random_perspective": {"distortion_scale": 0.5, "p": 0.5},
    "random_erasing": {"p": 0.5, "scale": (0.02, 0.33), "ratio": (0.3, 3.3)},
}

# Cross-validation configuration
num_folds = 5