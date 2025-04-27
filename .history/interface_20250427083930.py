# Replace MyCustomModel with the name of your model
from model import MyCustomModel as TheModel

# Replace train_model with the function inside train.py that runs the training loop
from train import train_model as the_trainer

# Replace predict with the function inside predict.py that runs inference
from predict import predict as the_predictor

# Replace CustomDataset with your custom Dataset class
from dataset import CustomDataset as TheDataset

# Replace custom_dataloader with your custom dataloader
from dataset import custom_dataloader as the_dataloader

# Import hyperparameters from config.py
from config import batchsize as the_batch_size
from config import epochs as total_epochs