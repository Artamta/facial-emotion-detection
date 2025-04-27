import torch
from model import MyCustomModel
from config import checkpoint_path, num_folds

def combine_models(model_name='VGG19', num_classes=7, output_path='./checkpoints/final_combined_model.pth'):
    """
    Combine the weights of all k-fold models into a single final model.

    Args:
        model_name (str): Name of the model architecture (e.g., 'VGG19').
        num_classes (int): Number of output classes.
        output_path (str): Path to save the combined model weights.
    """
    # Initialize the model
    model = MyCustomModel.get_model(model_name=model_name, num_classes=num_classes)
    combined_state_dict = None
    fold_count = 0

    # Iterate through all k-fold weight files
    for fold in range(1, num_folds + 1):
        fold_weight_path = f"{checkpoint_path}_fold{fold}.pth"
        print(f"[INFO] Loading weights from: {fold_weight_path}")
        checkpoint = torch.load(fold_weight_path, map_location='cpu')
        state_dict = checkpoint['net']

        # Add the weights to the combined state dict
        if combined_state_dict is None:
            combined_state_dict = {key: value.clone().float() for key, value in state_dict.items()}
        else:
            for key in combined_state_dict:
                combined_state_dict[key] += state_dict[key].float()

        fold_count += 1

    # Average the weights
    for key in combined_state_dict:
        combined_state_dict[key] /= fold_count

    # Save the combined weights
    torch.save({'net': combined_state_dict}, output_path)
    print(f"[INFO] Combined model saved to: {output_path}")

if __name__ == "__main__":
    combine_models()