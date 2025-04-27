# Replace MyCustomModel with your model class
from model import VGG as TheModel

# Replace train_model with your training function
from train import train_model as the_trainer

# Replace predict with your prediction function
from predict import predict as the_predictor

# Replace CustomDataset with your dataset class
from dataset import CustomDataset as TheDataset

# Replace custom_dataloader with your dataloader function
from dataset import custom_dataloader as the_dataloader

# Import hyperparameters from config.py
from config import batchsize as the_batch_size
from config import epochs as total_epochs