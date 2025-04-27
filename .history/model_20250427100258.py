from models.vgg import VGG
from models.resnet import ResNet18

class MyCustomModel:
    """
    A wrapper class to initialize and return the required model.
    """

    @staticmethod
    def get_model(model_name, num_classes=7):
        """
        Returns the specified model architecture.

        Args:
            model_name (str): Name of the model ('VGG19' or 'ResNet18').
            num_classes (int): Number of output classes. Default is 7 for CK+.

        Returns:
            torch.nn.Module: The initialized model.
        """
        if model_name == 'VGG19':
            print("[DEBUG] Initializing VGG19 model...")
            return VGG('VGG19')
        elif model_name == 'ResNet18':
            print("[DEBUG] Initializing ResNet18 model...")
            return ResNet18()
        else:
            raise ValueError(f"[ERROR] Unsupported model: {model_name}. Choose 'VGG19' or 'ResNet18'.")