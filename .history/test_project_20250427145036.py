import os
from config import (
    batch_size,
    epochs,
    learning_rate,
    checkpoint_path,
    dataset_path,
    resize_x,
    resize_y,
    input_channels,
    data_augmentation,
    num_folds,
)

def test_config():
    """
    Test the configuration values in config.py.
    """
    print("Testing configuration values...\n")

    # Check hyperparameters
    assert isinstance(batch_size, int) and batch_size > 0, "Invalid batch_size"
    print(f"Batch size: {batch_size} ✅")

    assert isinstance(epochs, int) and epochs > 0, "Invalid epochs"
    print(f"Epochs: {epochs} ✅")

    assert isinstance(learning_rate, float) and learning_rate > 0, "Invalid learning_rate"
    print(f"Learning rate: {learning_rate} ✅")

    # Check checkpoint path
    assert isinstance(checkpoint_path, str) and checkpoint_path.endswith(".pth"), "Invalid checkpoint_path"
    print(f"Checkpoint path: {checkpoint_path} ✅")

    # Check dataset path
    assert isinstance(dataset_path, str) and os.path.exists(dataset_path), "Dataset path does not exist"
    print(f"Dataset path: {dataset_path} ✅")

    # Check image dimensions
    assert isinstance(resize_x, int) and resize_x > 0, "Invalid resize_x"
    assert isinstance(resize_y, int) and resize_y > 0, "Invalid resize_y"
    assert isinstance(input_channels, int) and input_channels in [1, 3], "Invalid input_channels"
    print(f"Image dimensions: {resize_x}x{resize_y}, Channels: {input_channels} ✅")

    # Check data augmentation configuration
    assert isinstance(data_augmentation, dict), "Invalid data_augmentation configuration"
    print(f"Data augmentation configuration: {data_augmentation} ✅")

    # Check cross-validation configuration
    assert isinstance(num_folds, int) and num_folds > 1, "Invalid num_folds"
    print(f"Number of folds for cross-validation: {num_folds} ✅")

    print("\nAll configuration values are valid!")

if __name__ == "__main__":
    test_config()