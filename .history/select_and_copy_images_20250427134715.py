# Test script for interface.py
from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader, the_batch_size, total_epochs

print("Interface imports are working correctly.")
print(f"Batch size: {the_batch_size}, Total epochs: {total_epochs}")

# Instantiate the model
model = TheModel()
print("Model instantiated successfully.")

# Test dataset and dataloader
dataset = TheDataset(data_dir='./data')
dataloader = the_dataloader(data_dir='./data', batch_size=the_batch_size)
print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")