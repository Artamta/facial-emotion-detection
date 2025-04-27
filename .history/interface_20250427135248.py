# Replace MyCustomModel with the name of your model class
from model import MyCustomModel as TheModel

# Replace train_model with the function inside train.py that runs the training loop
from train import train_model as the_trainer

# Replace classify_images with the function inside predict.py that can be called to generate inference
from predict import classify_images as the_predictor

# Replace CustomDataset with your custom Dataset class
from dataset import CK as TheDataset

# Replace get_dataloader with your custom dataloader function
from dataset import get_dataloader as the_dataloader

# Replace batch_size and epochs with the corresponding variables in config.py
from config import batch_size as the_batch_size
from config import epochs as total_epochs